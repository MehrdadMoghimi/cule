#pragma once

#include <cule/config.hpp>

#include <cule/atari/actions.hpp>
#include <cule/atari/ale.hpp>
#include <cule/atari/controller.hpp>
#include <cule/atari/environment.hpp>
#include <cule/atari/functors.hpp>
#include <cule/atari/frame_state.hpp>
#include <cule/atari/joystick.hpp>
#include <cule/atari/m6502.hpp>
#include <cule/atari/paddles.hpp>
#include <cule/atari/palettes.hpp>
#include <cule/atari/png.hpp>
#include <cule/atari/preprocess.hpp>
#include <cule/atari/rom.hpp>

#include <agency/agency.hpp>

#include <random>

namespace cule
{
namespace atari
{
namespace dispatch
{

template<typename Environment,
         typename ExecutionPolicy,
         typename Wrapper>
void
reset(ExecutionPolicy&& policy,
      Wrapper& wrap,
      uint32_t* seedBuffer)
{
    agency::vector<uint32_t, agency::allocator<uint32_t>> rand_temp_buffer;

    if(seedBuffer == nullptr)
    {
        std::random_device rd;
        std::mt19937 rng(rd());
        rand_temp_buffer.resize(wrap.size(), 0);
        seedBuffer = rand_temp_buffer.data();

        std::generate_n(seedBuffer, wrap.size(), [&]{ return rng(); });
    }

    agency::bulk_invoke(policy(1),
                        initialize_functor<Environment>{},
                        wrap.rom_indices_ptr,
                        &wrap.cart,
                        wrap.cached_ram_ptr,
                        wrap.cached_states_ptr,
                        wrap.cached_frame_states_ptr,
                        wrap.rand_states_ptr,
                        seedBuffer);

    // Boot the base state, then keep stepping (bounded) while the game still
    // reports terminal: several games (e.g. ms_pacman, space_invaders) need
    // extra frames after the console RESET before their lives/score RAM is
    // initialized, and caching those pre-boot states leaves every reset
    // permanently terminal. Games that boot clean take no extra frames.
    //
    // Count completed *frames*, not act() calls: boot sequences with heavy
    // TIA traffic (e.g. qbert) split one frame across many chunked act()
    // calls when the update buffer fills, and counting calls used to leave
    // pre-boot states in the cache (the game then never started).
    const int32_t max_boot_frames = 16 * ENV_BASE_FRAMES;
    const size_t max_boot_iters = 64 * size_t(max_boot_frames);
    bool game_booting = true;
    for (size_t i = 0; i < max_boot_iters; i++)
    {
        const int32_t frame = Environment::getFrameNumber(wrap.cached_states_ptr[0]);
        if (frame >= max_boot_frames)
            break;
        if ((frame >= ENV_BASE_FRAMES) && !game_booting)
            break;
        agency::bulk_invoke(policy(1),
                            step_functor<Environment>{},
                            true,
                            wrap.cached_states_ptr,
                            wrap.cached_tia_update_ptr,
                            nullptr,
                            nullptr,
                            nullptr);
        // read the per-game terminal state recomputed by the emulation step,
        // before the preprocess pass below clears it for frame rendering
        game_booting = wrap.cached_states_ptr[0].tiaFlags[FLAG_ALE_TERMINAL];
        agency::bulk_invoke(policy(1),
                            preprocess_functor<Environment>{},
                            true,
                            wrap.cached_tia_update_ptr,
                            nullptr,
                            wrap.cached_states_ptr,
                            wrap.cache_index_ptr,
                            wrap.cached_frame_states_ptr,
                            nullptr);
    }

    for (size_t i = 1; i < wrap.noop_reset_steps; i++)
    {
        wrap.cached_states_ptr[i] = wrap.cached_states_ptr[i - 1];
        wrap.cached_frame_states_ptr[i] = wrap.cached_frame_states_ptr[i - 1];

        wrap.cached_states_ptr[i].ram = (uint32_t*) (wrap.cached_ram_ptr + (wrap.cart.ram_size() * i));
        std::copy(wrap.cached_states_ptr[i - 1].ram,
                  wrap.cached_states_ptr[i - 1].ram + (wrap.cart.ram_size() / sizeof(uint32_t)),
                  wrap.cached_states_ptr[i].ram);

        agency::bulk_invoke(policy(1),
                            step_functor<Environment>{},
                            true,
                            wrap.cached_states_ptr + i,
                            wrap.cached_tia_update_ptr + (i * ENV_UPDATE_SIZE),
                            nullptr,
                            nullptr,
                            nullptr);
        agency::bulk_invoke(policy(1),
                            preprocess_functor<Environment>{},
                            true,
                            wrap.cached_tia_update_ptr + (i * ENV_UPDATE_SIZE),
                            nullptr,
                            wrap.cached_states_ptr + i,
                            wrap.cache_index_ptr + i,
                            wrap.cached_frame_states_ptr + i,
                            nullptr);
    }

    // Sync each cached state's score to what its RAM actually decodes to.
    // Games whose start score is nonzero (e.g. pitfall boots with 2000
    // points) otherwise emit a spurious reward on the first step after
    // every reset (ALE syncs m_score during reset the same way).
    for (size_t i = 0; i < wrap.noop_reset_steps; i++)
    {
        wrap.cached_states_ptr[i].score =
            Environment::ALE_t::getScore(wrap.cached_states_ptr[i]);
    }

    for (size_t i = 0; i < wrap.size(); i++)
    {
        const size_t index = rand() % wrap.noop_reset_steps;
        wrap.states_ptr[i] = wrap.cached_states_ptr[index];
        wrap.states_ptr[i].ram = (uint32_t *) (wrap.ram_ptr + (wrap.cart.ram_size() * i));
        std::copy(wrap.cached_states_ptr[index].ram,
                  wrap.cached_states_ptr[index].ram + (wrap.cart.ram_size() / sizeof(uint32_t)),
                  wrap.states_ptr[i].ram);
        wrap.frame_states_ptr[i] = wrap.cached_frame_states_ptr[index];
    }
}

template<typename Environment,
         typename ExecutionPolicy,
         typename Wrapper>
void
reset_states(ExecutionPolicy&& policy,
             Wrapper& wrap)
{
    agency::bulk_invoke(policy(wrap.size()),
                        reset_functor<Environment>{},
                        wrap.cart.ram_size(),
                        wrap.noop_reset_steps,
                        wrap.states_ptr,
                        wrap.cached_states_ptr,
                        wrap.cached_ram_ptr,
                        wrap.frame_states_ptr,
                        wrap.cached_frame_states_ptr,
                        wrap.cache_index_ptr,
                        wrap.rand_states_ptr);
}

template<typename ExecutionPolicy,
         typename Wrapper>
void
get_states(ExecutionPolicy&& policy,
           Wrapper& wrap,
           const size_t num_states,
           const int32_t* indices,
           typename Wrapper::State_t* output_states,
           frame_state* output_frame_states,
           uint8_t*)
{
    agency::bulk_invoke(policy(num_states),
                        get_states_functor{},
                        indices,
                        wrap.states_ptr,
                        wrap.frame_states_ptr,
                        output_states,
                        output_frame_states);
}

template<typename ExecutionPolicy,
         typename Wrapper>
void
set_states(ExecutionPolicy&& policy,
           Wrapper& wrap,
           const size_t num_states,
           const int32_t* indices,
           const typename Wrapper::State_t* input_states,
           const frame_state* input_frame_states,
           const uint8_t*)
{
    agency::bulk_invoke(policy(num_states),
                        set_states_functor{},
                        indices,
                        wrap.states_ptr,
                        wrap.frame_states_ptr,
                        input_states,
                        input_frame_states);
}

template<typename Environment,
         typename ExecutionPolicy,
         typename Wrapper>
void
step(ExecutionPolicy&& policy,
     Wrapper& wrap,
     const bool fire_reset,
     const Action* playerABuffer,
     const Action* playerBBuffer,
     bool* doneBuffer)
{
    agency::bulk_invoke(policy(wrap.size()),
                        step_functor<Environment>{},
                        fire_reset,
                        wrap.states_ptr,
                        wrap.tia_update_ptr,
                        (Action*) playerABuffer,
                        (Action*) playerBBuffer,
                        doneBuffer);
}

template<typename Environment,
         typename ExecutionPolicy,
         typename Wrapper>
void
get_data(ExecutionPolicy&& policy,
         Wrapper& wrap,
         const bool episodic_life,
         bool* doneBuffer,
         float* rewardsBuffer,
         int32_t* livesBuffer)
{
    agency::bulk_invoke(policy(wrap.size()),
                        get_data_functor<Environment>{},
                        episodic_life,
                        wrap.states_ptr,
                        doneBuffer,
                        rewardsBuffer,
                        livesBuffer);
}

template<typename Environment,
         typename ExecutionPolicy,
         typename Wrapper>
void
preprocess(ExecutionPolicy&& policy,
           Wrapper& wrap,
           const bool last_frame,
           const uint32_t* tiaBuffer,
           uint8_t* frameBuffer)
{
    agency::bulk_invoke(policy(wrap.size()),
                        preprocess_functor<Environment>{},
                        last_frame,
                        tiaBuffer,
                        wrap.cached_tia_update_ptr,
                        wrap.states_ptr,
                        wrap.cache_index_ptr,
                        wrap.frame_states_ptr,
                        frameBuffer);
}

template<typename ExecutionPolicy,
         typename Wrapper>
void
generate_frames(ExecutionPolicy&& policy,
                Wrapper& wrap,
                const bool rescale,
                const size_t num_channels,
                uint8_t* imageBuffer)
{
    agency::bulk_invoke(policy(wrap.size()),
                        generate_frame_functor{},
                        num_channels,
                        wrap.cart.screen_height(),
                        rescale,
                        wrap.frame_ptr,
                        imageBuffer);
}

template<typename ExecutionPolicy,
         typename Wrapper>
void
generate_random_actions(ExecutionPolicy&& policy,
                        Wrapper& wrap,
                        Action* actionsBuffer,
                        const size_t N)
{
    const size_t num_entries = N == 0 ? wrap.size() : N;

    agency::bulk_invoke(policy(wrap.size()),
                        random_actions_functor{},
                        wrap.cart.minimal_actions().size(),
                        num_entries,
                        wrap.minimal_actions_ptr,
                        wrap.rand_states_ptr,
                        actionsBuffer);
}

template<typename ExecutionPolicy,
         typename Wrapper>
void
save_images(ExecutionPolicy&& policy,
            Wrapper& wrap,
            const bool rescale,
            const size_t num_channels,
            const size_t frame_index,
            const uint8_t* imageBuffer)
{
    agency::bulk_invoke(policy(wrap.size()),
                        png_functor{},
                        frame_index,
                        num_channels,
                        wrap.cart.screen_height(),
                        rescale,
                        imageBuffer);
}

} // end namespace dispatch
} // end namespace atari
} // end namespace cule

#ifdef __CUDACC__
#include <cule/atari/cuda/dispatch.hpp>
#endif
