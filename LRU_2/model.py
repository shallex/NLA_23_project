from functools import partial
import jax
import jax.numpy as jnp
import torch
import numpy as np
import math
import torch
import torch.nn as nn


parallel_scan = jax.lax.associative_scan



def binary_operator_diag(q_i, q_j):
    """Binary operator for parallel scan of linear recurrence"""
    A_i, b_i = q_i
    A_j, b_j = q_j
    return A_j * A_i, A_j * b_i + b_j


def matrix_init(shape, dtype=torch.float32, normalization=1):
    print(shape)
    print(type(shape))
    return torch.randn(shape) / normalization


def nu_init(key, shape, r_min, r_max, dtype=jnp.float_):
    u = jax.random.uniform(key=key, shape=shape, dtype=dtype)
    return jnp.log(-0.5 * jnp.log(u * (r_max**2 - r_min**2) + r_min**2))


def theta_init(key, shape, max_phase, dtype=jnp.float_):
    u = jax.random.uniform(key, shape=shape, dtype=dtype)
    return jnp.log(max_phase * u)


def gamma_log_init(key, lamb):
    nu, theta = lamb
    diag_lambda = jnp.exp(-jnp.exp(nu) + 1j * jnp.exp(theta))
    return jnp.log(jnp.sqrt(1 - jnp.abs(diag_lambda) ** 2))


class LRU(torch.nn.Module):
    def __init__(self,in_features,out_features, state_features, rmin=0, rmax=1,max_phase=6.283):
        super().__init__()
        self.out_features=out_features
        self.D=nn.Parameter(torch.randn([out_features,in_features])/math.sqrt(in_features))
        u1=torch.rand(state_features)
        u2=torch.rand(state_features)
        self.nu_log= nn.Parameter(torch.log(-0.5*torch.log(u1*(rmax+rmin)*(rmax-rmin) + rmin**2)))
        self.theta_log= nn.Parameter(torch.log(max_phase*u2))
        Lambda_mod=torch.exp(-torch.exp(self.nu_log))
        self.gamma_log=nn.Parameter(torch.log(torch.sqrt(torch.ones_like(Lambda_mod)-torch.square(Lambda_mod))))
        B_re=torch.randn([state_features,in_features])/math.sqrt(2*in_features)
        B_im=torch.randn([state_features,in_features])/math.sqrt(2*in_features)
        self.B=nn.Parameter(torch.complex(B_re,B_im))
        C_re=torch.randn([out_features,state_features])/math.sqrt(state_features)
        C_im=torch.randn([out_features,state_features])/math.sqrt(state_features)
        self.C=nn.Parameter(torch.complex(C_re,C_im))
        self.state=torch.complex(torch.zeros(state_features),torch.zeros(state_features))

    def forward(self, input,state=None):
        self.state=self.state.to(self.B.device) if state==None else state
        Lambda_mod=torch.exp(-torch.exp(self.nu_log))
        Lambda_re=Lambda_mod*torch.cos(torch.exp(self.theta_log))
        Lambda_im=Lambda_mod*torch.sin(torch.exp(self.theta_log))
        Lambda=torch.complex(Lambda_re,Lambda_im)
        Lambda=Lambda.to(self.state.device)
        gammas=torch.exp(self.gamma_log).unsqueeze(-1).to(self.B.device)
        gammas=gammas.to(self.state.device)
        output=torch.empty([i for i in input.shape[:-1]] +[self.out_features],device=self.B.device)
        #Handle input of (Batches, Seq_length, Input size)


        print(f"LRU input: {input.shape}")
        if input.dim()==3:
            for i, batch in enumerate(input):
                out_seq=torch.empty(input.shape[1], self.out_features, device=torch.device("cuda"))
                for j, step in enumerate(batch):
                    # print(f"{out_seq.device=}")
                    # print(f"{step.device=}")
                    self.state=(Lambda * self.state + gammas * self.B @ step.to(dtype=self.B.dtype))
                    # print(f"{self.state.device=}")
                    out_step= (self.C@self.state).real + self.D@step
                    # print(f"{out_step.device=}")
                    out_seq[j]=out_step
                    # print(f"{out_seq.device=}")
                self.state=torch.complex(torch.zeros_like(self.state.real),torch.zeros_like(self.state.real))
                output[i]=out_seq
            print(f"LRU output: {output.shape}")
        #Handle input of (Seq_length, Input size)
        if input.dim()==2:
            for i,step in enumerate(input):
                self.state=(Lambda*self.state + gammas* self.B@step.to(dtype= self.B.dtype))
                out_step= (self.C@self.state).real + self.D@step
                output[i]=out_step
            self.state=torch.complex(torch.zeros_like(self.state.real),torch.zeros_like(self.state.real))
        return output


class SequenceLayer(torch.nn.Module):
    """Single layer, with one LRU module, GLU, dropout and batch/layer norm"""
    def __init__(self, lru,
            d_model,
            dropout,
            training,
            norm):
        super().__init__()
        self.lru = lru
        self.d_model = d_model

        self.dropout  = dropout
        self.training  = training 
        self.norm = norm

        """Initializes the ssm, layer norm and dropout"""
        self.seq = self.lru
        self.out1 = nn.Linear(self.d_model, self.d_model)
        self.out2 = nn.Linear(self.d_model, self.d_model)
        if self.norm in ["layer"]:
            self.normalization = nn.LayerNorm()
        else:
            self.normalization = nn.functional.batch_norm
        
    def forward(self, inputs):
        # x = self.normalization(inputs)  # pre normalization
        x = self.seq(inputs)  # call LRU
        # x = self.drop(nn.gelu(x))
        x = self.out1(x) * torch.sigmoid(self.out2(x))  # GLU
        # x = self.drop(x)
        return inputs + x  # skip connection


class StackedEncoderModel(torch.nn.Module):
    """Encoder containing several SequenceLayer"""
    def __init__(self, lru,
            d_model,
            n_layers,
            dropout,
            training,
            norm):
        super().__init__()
        self.lru = lru
        self.d_model = d_model
        self.n_layers = n_layers
        self.dropout  = dropout
        self.training  = training 
        self.norm = norm
         
        self.encoder = nn.Linear(1, self.d_model)
        self.layers = nn.ModuleList([
            SequenceLayer(
                lru=self.lru,
                d_model=self.d_model,
                dropout=self.dropout,
                training=self.training,
                norm=self.norm,
            )
            for _ in range(self.n_layers)
        ])

    def forward(self, inputs):

        # print(inputs.shape)

        x = self.encoder(inputs)  # embed input in latent space
        print(f"after linear in StackedEncoderModel: {x.shape}")
        # print(x.shape)
        for layer in self.layers:
            x = layer(x)  # apply each layer
        return x


class ClassificationModel(torch.nn.Module):
    """Stacked encoder with pooling and softmax"""
    def __init__(self, lru: nn.Module,
                        d_output: int,
                        d_model: int,
                        n_layers: int,
                        dropout: float = 0.0,
                        training: bool = True,
                        pooling: str = "mean",  # pooling mode
                        norm: str = "batch",  # type of normaliztion
                        multidim: int = 1):
        
        super().__init__()
        self.pooling = pooling
        # lru: nn.Module
        # d_output: int
        # d_model: int
        # n_layers: int
        # dropout: float = 0.0
        # training: bool = True
        # pooling: str = "mean"  # pooling mode
        # norm: str = "batch"  # type of normaliztion
        # multidim: int = 1  # number of outputs
        self.multidim = multidim
        self.d_output = d_output

        self.encoder = StackedEncoderModel(
            lru=lru,
            d_model=d_model,
            n_layers=n_layers,
            dropout=dropout,
            training=training,
            norm=norm,
        )
        self.decoder = nn.Linear(d_model, d_output)
        

    def forward(self, x):
        print("Start model")
        print(x.shape)
        x = self.encoder(x)
        print("need for time")
        print(x.shape)
        if self.pooling in ["mean"]:
            print("use mean")
            x = torch.mean(x, axis=1)  # mean pooling across time
        elif self.pooling in ["last"]:
            x = x[-1]  # just take last
        elif self.pooling in ["none"]:
            x = x  # do not pool at all
        x = self.decoder(x)
        if self.multidim > 1:
            x = x.reshape(-1, self.d_output, self.multidim)

        print("after decoder")
        print(x.shape)
        return torch.nn.functional.log_softmax(x, dim=-1)