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
from curriculum_foresight.policy.subroutines import CallEligibility, subroutine_action_distribution


class SubroutineActorCritic(nn.Module):
    env_id: str
    num_primitive_actions: int
    num_goals: int
    max_subroutine_steps: int
    obs_emb_dim: int
    hidden_dim: int
    num_attn_heads: int
    qkv_features: int
    num_layers_in_transformer: int
    gating_bias: float
    head_hidden_dim: int

    def setup(self):
        self.input_encoder = PolicyInputEncoder(env_id=self.env_id, obs_emb_dim=self.obs_emb_dim)
        self.transformer = GatedTransformerXL(
            hidden_dim=self.hidden_dim,
            num_heads=self.num_attn_heads,
            qkv_features=self.qkv_features,
            num_layers=self.num_layers_in_transformer,
            gating_bias=self.gating_bias,
        )
        self.actor_linear1 = nn.Dense(self.head_hidden_dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))
        self.actor_linear2 = nn.Dense(self.head_hidden_dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))
        self.actor_out = nn.Dense(
            self.num_primitive_actions + self.num_goals, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )
        self.critic_linear1 = nn.Dense(
            self.head_hidden_dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )
        self.critic_linear2 = nn.Dense(
            self.head_hidden_dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )
        self.critic_out = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))

    def _heads(
        self,
        features: jax.Array,
        assigned_goal_embeddings: jax.Array,
        active_goal_embeddings: jax.Array,
        remaining_subroutine_steps: jax.Array,
        eligibility: CallEligibility,
    ) -> tuple[distrax.Categorical, jax.Array]:
        actor_input = jnp.concatenate((features, active_goal_embeddings), axis=-1)
        actor = nn.relu(self.actor_linear1(actor_input))
        actor = nn.relu(self.actor_linear2(actor))
        distribution = subroutine_action_distribution(
            self.actor_out(actor), eligibility, num_primitive_actions=self.num_primitive_actions
        )

        subroutine_time_fraction = (
            remaining_subroutine_steps[..., None].astype(features.dtype) / self.max_subroutine_steps
        )
        critic_input = jnp.concatenate(
            (features, assigned_goal_embeddings, active_goal_embeddings, subroutine_time_fraction), axis=-1
        )
        critic = nn.relu(self.critic_linear1(critic_input))
        critic = nn.relu(self.critic_linear2(critic))
        values = jnp.squeeze(self.critic_out(critic), axis=-1)
        return distribution, values

    def __call__(
        self,
        memories: jax.Array,
        inputs: Mapping[str, jax.Array],
        mask: jax.Array,
        assigned_goal_embeddings: jax.Array,
        active_goal_embeddings: jax.Array,
        remaining_subroutine_steps: jax.Array,
        eligibility: CallEligibility,
    ) -> tuple[distrax.Categorical, jax.Array]:
        encoded_inputs = self.input_encoder(inputs)[:, 0, :]
        features = self.transformer(memories, encoded_inputs, mask)
        return self._heads(
            features, assigned_goal_embeddings, active_goal_embeddings, remaining_subroutine_steps, eligibility
        )

    def forward_step(
        self,
        memories: jax.Array,
        inputs: Mapping[str, jax.Array],
        mask: jax.Array,
        assigned_goal_embeddings: jax.Array,
        active_goal_embeddings: jax.Array,
        remaining_subroutine_steps: jax.Array,
        eligibility: CallEligibility,
    ) -> tuple[distrax.Categorical, jax.Array, jax.Array]:
        encoded_inputs = self.input_encoder(inputs)[:, 0, :]
        features, memory_out = self.transformer.forward_step(memories, encoded_inputs, mask)
        distribution, values = self._heads(
            features, assigned_goal_embeddings, active_goal_embeddings, remaining_subroutine_steps, eligibility
        )
        return distribution, values, memory_out

    def forward_sequence(
        self,
        memories: jax.Array,
        inputs: Mapping[str, jax.Array],
        mask: jax.Array,
        assigned_goal_embeddings: jax.Array,
        active_goal_embeddings: jax.Array,
        remaining_subroutine_steps: jax.Array,
        eligibility: CallEligibility,
    ) -> tuple[distrax.Categorical, jax.Array]:
        encoded_inputs = self.input_encoder(inputs)
        features = self.transformer.forward_sequence(memories, encoded_inputs, mask)
        assigned_goal_embeddings = jnp.broadcast_to(
            assigned_goal_embeddings[:, None, :], (*features.shape[:2], assigned_goal_embeddings.shape[-1])
        )
        return self._heads(
            features, assigned_goal_embeddings, active_goal_embeddings, remaining_subroutine_steps, eligibility
        )
