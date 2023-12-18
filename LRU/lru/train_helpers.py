from functools import partial
import jax
import jax.numpy as jnp
from jax.nn import one_hot
from tqdm import tqdm
from flax.training import train_state
import optax
from typing import Any


def linear_warmup(step, base_lr, end_step, lr_min=None):
    return base_lr * (step + 1) / end_step


def cosine_annealing(step, base_lr, end_step, lr_min=1e-6):
    # https://github.com/deepmind/optax/blob/master/optax/_src/schedule.py#L207#L240
    count = jnp.minimum(step, end_step)
    cosine_decay = 0.5 * (1 + jnp.cos(jnp.pi * count / end_step))
    decayed = (base_lr - lr_min) * cosine_decay + lr_min
    return decayed


def constant_lr(step, base_lr, end_step, lr_min=None):
    return base_lr


def update_learning_rate_per_step(lr_params, state):
    decay_function, ssm_lr, lr, step, end_step, lr_min = lr_params

    # Get decayed value
    lr_val = decay_function(step, lr, end_step, lr_min)
    ssm_lr_val = decay_function(step, ssm_lr, end_step, lr_min)
    step += 1

    # Update state
    state.opt_state.inner_states["regular"].inner_state.hyperparams["learning_rate"] = jnp.array(
        lr_val, dtype=jnp.float32
    )
    state.opt_state.inner_states["ssm"].inner_state.hyperparams["learning_rate"] = jnp.array(
        ssm_lr_val, dtype=jnp.float32
    )
    return state, step


def reduce_lr_on_plateau(input, factor=0.2, patience=20, lr_min=1e-6):
    lr, ssm_lr, count, new_acc, opt_acc = input
    if new_acc > opt_acc:
        count = 0
        opt_acc = new_acc
    else:
        count += 1

    if count > patience:
        lr = factor * lr
        ssm_lr = factor * ssm_lr
        count = 0

    if lr < lr_min:
        lr = lr_min
    if ssm_lr < lr_min:
        ssm_lr = lr_min

    return lr, ssm_lr, count, opt_acc


def map_nested_fn(fn):
    """
    Recursively apply `fn to the key-value pairs of a nested dict / pytree.
    We use this for some of the optax definitions below.
    """

    def map_fn(nested_dict):
        return {k: (map_fn(v) if hasattr(v, "keys") else fn(k, v)) for k, v in nested_dict.items()}

    return map_fn


def create_train_state(model_cls, rng, in_dim, batch_size, seq_len, weight_decay, norm, ssm_lr, lr):
    """Initializes the training state using optax"""

    dummy_input = jnp.ones((batch_size, seq_len, in_dim))
    model = model_cls(training=True)
    init_rng, dropout_rng = jax.random.split(rng, num=2)
    variables = model.init({"params": init_rng, "dropout": dropout_rng}, dummy_input)
    if norm in ["batch"]:
        params = variables["params"].unfreeze()
        batch_stats = variables["batch_stats"]
    else:
        params = variables["params"].unfreeze()  # NOTE: unfreeze is for optax

    # Smaller lr and no weight decay for lambda, gamma and B
    ssm_fn = map_nested_fn(
        lambda k, _: "ssm"
        if k in ["nu_log", "theta_log", "gamma_log", "B_re", "B_im"]
        else "regular"
    )
    tx = optax.multi_transform(
        {
            "ssm": optax.inject_hyperparams(optax.adam)(learning_rate=ssm_lr),
            "regular": optax.inject_hyperparams(optax.adamw)(
                learning_rate=lr, weight_decay=weight_decay
            ),
        },
        ssm_fn,
    )

    def fn_is_complex(x):
        return x.dtype in [jnp.complex64, jnp.complex128]

    param_sizes = map_nested_fn(lambda k, param: param.size * (2 if fn_is_complex(param) else 1))(
        params
    )
    print(f"[*] Trainable Parameters: {sum(jax.tree_util.tree_leaves(param_sizes))}")

    if norm in ["batch"]:

        class TrainState(train_state.TrainState):
            batch_stats: Any

        return TrainState.create(
            apply_fn=model.apply, params=params, tx=tx, batch_stats=batch_stats
        )
    else:
        return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)


@jax.vmap
def batched_average_mask(a, mask):
    """Average of a by sum of values of mask"""
    return a / jnp.sum(mask)


@jax.vmap
def create_mask(x, length):
    L = x.shape[0]
    mask = (jnp.arange(L) >= length[0]) * (jnp.arange(L) < length[1])
    return mask


@partial(jnp.vectorize, signature="(c),()->()")
def cross_entropy_loss(logits, label):
    one_hot_label = jax.nn.one_hot(label, num_classes=logits.shape[0])
    return -jnp.sum(one_hot_label * logits)


@partial(jnp.vectorize, signature="(c),()->()")
def compute_accuracy(logits, label):
    return jnp.argmax(logits) == label


def compute_accuracies(logits, labels, masks):
    if len(logits.shape) == 4:
        return jnp.sum(
            batched_average_mask(masks * compute_accuracy(logits, labels).mean(axis=-1), masks),
            axis=-1,
        )
    elif len(logits.shape) == 2:
        return jnp.mean(compute_accuracy(logits, labels))


def loss_fn(logits, labels, masks):
    """
    Pick the desired loss depending on the shape of the logits (and therefore the task)
    """
    if len(logits.shape) == 2:  # for classification tasks
        losses = cross_entropy_loss(logits, labels)
    if len(logits.shape) == 4:  # for tasks with multidimensional dense targets
        losses = masks * cross_entropy_loss(logits, labels).mean(axis=-1)
        losses = batched_average_mask(losses, masks)  # average over time
    return jnp.mean(losses)


def prep_batch(batch, seq_len, in_dim):
    """Take a batch and convert it to a standard x/y format"""
    if len(batch) == 2:
        inputs, targets = batch
        aux_data = {}
    elif len(batch) == 3:
        inputs, targets, aux_data = batch
    else:
        raise RuntimeError("Unhandled data type. ")

    inputs = jnp.array(inputs.numpy()).astype(float)  # convert to jax
    targets = jnp.array(targets.numpy())  # convert to jax
    lengths = aux_data.get("lengths", None)  # get lengths from aux if it is there.

    # Make all batches have same sequence length
    num_pad = seq_len - inputs.shape[1]
    if num_pad > 0:
        inputs = jnp.pad(inputs, ((0, 0), (0, num_pad)), "constant", constant_values=(0,))

    # Inputs size is [n_batch, seq_len] or [n_batch, seq_len, in_dim].
    # If there are not three dimensions and trailing dimension is not equal to in_dim then
    # transform into one-hot.  This should be a fairly reliable fix.
    if (inputs.ndim < 3) and (inputs.shape[-1] != in_dim):
        inputs = one_hot(inputs, in_dim)

    if lengths is not None:
        lengths = jnp.array(lengths)
        if len(lengths.shape) == 1:  # If lengths only give last
            lengths = jnp.stack([jnp.zeros((inputs.shape[0],)), lengths], axis=1)
        masks = create_mask(inputs, lengths)
    else:
        masks = jnp.ones((inputs.shape[0], inputs.shape[1]))

    return inputs, targets, masks


@partial(jax.jit, static_argnums=(5, 6))
def train_step(state, rng, inputs, labels, masks, model, norm):
    """Performs a single training step given a batch of data"""

    def _loss(params):
        if norm in ["batch"]:
            p = {"params": params, "batch_stats": state.batch_stats}
            logits, vars = model.apply(p, inputs, rngs={"dropout": rng}, mutable=["batch_stats"])
        else:
            p = {"params": params}
            vars = None
            logits = model.apply(p, inputs, rngs={"dropout": rng})
        return loss_fn(logits, labels, masks), vars

    (loss, vars), grads = jax.value_and_grad(_loss, has_aux=True)(state.params)

    if norm in ["batch"]:
        state = state.apply_gradients(grads=grads, batch_stats=vars["batch_stats"])
    else:
        state = state.apply_gradients(grads=grads)

    return state, loss


def train_epoch(state, rng, model, trainloader, seq_len, in_dim, norm, lr_params):
    """
    Training function for an epoch that loops over batches.
    """
    model = model(training=True)  # model in training mode
    batch_losses = []
    decay_function, ssm_lr, lr, step, end_step, lr_min = lr_params

    for batch in tqdm(trainloader):
        inputs, labels, masks = prep_batch(batch, seq_len, in_dim)
        rng, drop_rng = jax.random.split(rng)
        state, loss = train_step(state, drop_rng, inputs, labels, masks, model, norm)
        batch_losses.append(loss)  # log loss value

        lr_params = (decay_function, ssm_lr, lr, step, end_step, lr_min)
        state, step = update_learning_rate_per_step(lr_params, state)

    # Return average loss over batches
    return state, jnp.mean(jnp.array(batch_losses)), step


@partial(jax.jit, static_argnums=(4, 5))
def eval_step(inputs, labels, masks, state, model, norm):
    if norm == "batch":
        logits = model.apply({"params": state.params, "batch_stats": state.batch_stats}, inputs)
    else:
        logits = model.apply({"params": state.params}, inputs)
    losses = loss_fn(logits, labels, masks)
    accs = compute_accuracies(logits, labels, masks)
    return jnp.mean(losses), accs, logits


def validate(state, model, testloader, seq_len, in_dim, norm):
    """Validation function that loops over batches"""
    model = model(training=False)
    losses, accuracies = jnp.array([]), jnp.array([])

    for batch in tqdm(testloader):
        inputs, labels, masks = prep_batch(batch, seq_len, in_dim)
        loss, acc, logits = eval_step(inputs, labels, masks, state, model, norm)
        losses = jnp.append(losses, loss)
        accuracies = jnp.append(accuracies, acc)
    return jnp.mean(losses), jnp.mean(accuracies)
