from __future__ import annotations

from collections.abc import Mapping

import distrax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from flax.linen.initializers import constant, orthogonal

from curriculum_foresight.networks.encoders import PolicyInputEncoder
from curriculum_foresight.networks.gated_transformer_xl import GatedTransformerXL


class PrimitiveActorCritic(nn.Module):
    env_id: str
    num_actions: int

    obs_emb_dim: int

    hidden_dim: int
    num_attn_heads: int
    qkv_features: int
    num_layers_in_transformer: int
    gating_bias: float

    head_hidden_dim: int

    transformer_input_fusion_hidden_dim: int = 512

    def setup(self):
        self.input_encoder = PolicyInputEncoder(env_id=self.env_id, obs_emb_dim=self.obs_emb_dim)

        self.transformer = GatedTransformerXL(
            hidden_dim=self.hidden_dim,
            num_heads=self.num_attn_heads,
            qkv_features=self.qkv_features,
            num_layers=self.num_layers_in_transformer,
            gating_bias=self.gating_bias,
        )
        self.transformer_input_fusion = nn.Dense(
            self.transformer_input_fusion_hidden_dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )

        self.actor_linear1 = nn.Dense(self.head_hidden_dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))
        self.actor_linear2 = nn.Dense(self.head_hidden_dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))
        self.actor_out = nn.Dense(self.num_actions, kernel_init=orthogonal(0.01), bias_init=constant(0.0))

        self.critic_linear1 = nn.Dense(
            self.head_hidden_dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )
        self.critic_linear2 = nn.Dense(
            self.head_hidden_dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )
        self.critic_out = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))

    def __call__(
        self, memories: jax.Array, inputs: Mapping[str, jax.Array], mask: jax.Array, goal_embeddings: jax.Array
    ) -> tuple[distrax.Categorical, jax.Array]:
        encoded_inputs = self.input_encoder(inputs)[:, 0, :]
        transformer_inputs = jnp.concatenate((encoded_inputs, goal_embeddings), axis=-1)
        transformer_inputs = nn.gelu(self.transformer_input_fusion(transformer_inputs))
        features = self.transformer(memories, transformer_inputs, mask)

        actor_input = jnp.concatenate((features, goal_embeddings), axis=-1)
        actor = self.actor_linear1(actor_input)
        actor = nn.relu(actor)
        actor = self.actor_linear2(actor)
        actor = nn.relu(actor)
        actor_logits = self.actor_out(actor)
        pi = distrax.Categorical(logits=actor_logits)

        critic_input = jnp.concatenate((features, goal_embeddings), axis=-1)
        critic = self.critic_linear1(critic_input)
        critic = nn.relu(critic)
        critic = self.critic_linear2(critic)
        critic = nn.relu(critic)
        values = jnp.squeeze(self.critic_out(critic), axis=-1)

        return pi, values

    def forward_step(
        self, memories: jax.Array, inputs: Mapping[str, jax.Array], mask: jax.Array, goal_embeddings: jax.Array
    ) -> tuple[distrax.Categorical, jax.Array, jax.Array]:
        encoded_inputs = self.input_encoder(inputs)[:, 0, :]
        transformer_inputs = jnp.concatenate((encoded_inputs, goal_embeddings), axis=-1)
        transformer_inputs = nn.gelu(self.transformer_input_fusion(transformer_inputs))
        features, memory_out = self.transformer.forward_step(memories, transformer_inputs, mask)

        actor_input = jnp.concatenate((features, goal_embeddings), axis=-1)
        actor = self.actor_linear1(actor_input)
        actor = nn.relu(actor)
        actor = self.actor_linear2(actor)
        actor = nn.relu(actor)
        actor_logits = self.actor_out(actor)
        pi = distrax.Categorical(logits=actor_logits)

        critic_input = jnp.concatenate((features, goal_embeddings), axis=-1)
        critic = self.critic_linear1(critic_input)
        critic = nn.relu(critic)
        critic = self.critic_linear2(critic)
        critic = nn.relu(critic)
        values = jnp.squeeze(self.critic_out(critic), axis=-1)

        return pi, values, memory_out

    def forward_sequence(
        self, memories: jax.Array, inputs: Mapping[str, jax.Array], mask: jax.Array, goal_embeddings: jax.Array
    ) -> tuple[distrax.Categorical, jax.Array]:
        encoded_inputs = self.input_encoder(inputs)
        goal_embeddings_over_time = jnp.broadcast_to(
            goal_embeddings[:, None, :], (goal_embeddings.shape[0], encoded_inputs.shape[1], goal_embeddings.shape[1])
        )

        transformer_inputs = jnp.concatenate((encoded_inputs, goal_embeddings_over_time), axis=-1)
        transformer_inputs = nn.gelu(self.transformer_input_fusion(transformer_inputs))
        features = self.transformer.forward_sequence(memories, transformer_inputs, mask)

        actor_input = jnp.concatenate((features, goal_embeddings_over_time), axis=-1)
        actor = self.actor_linear1(actor_input)
        actor = nn.relu(actor)
        actor = self.actor_linear2(actor)
        actor = nn.relu(actor)
        actor_logits = self.actor_out(actor)
        pi = distrax.Categorical(logits=actor_logits)

        critic_input = jnp.concatenate((features, goal_embeddings_over_time), axis=-1)
        critic = self.critic_linear1(critic_input)
        critic = nn.relu(critic)
        critic = self.critic_linear2(critic)
        critic = nn.relu(critic)
        values = jnp.squeeze(self.critic_out(critic), axis=-1)

        return pi, values
