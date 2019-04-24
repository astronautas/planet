from atari_wrappers import MaxAndSkipEnv, EpisodicLifeEnv, ClipRewardEnv

def wrap_vizdoom(env, episode_logging_file, episode_life=True, clip_rewards=True, max_and_skip=True):
    if episode_life:
        env = EpisodicLifeEnv(env, episode_logging_file=episode_logging_file)

    if clip_rewards:
        env = ClipRewardEnv(env)

    if max_and_skip:
        env = MaxAndSkipEnv(env)

    return env