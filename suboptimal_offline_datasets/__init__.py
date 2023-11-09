import os

from gym.envs.registration import register
from d4rl.gym_mujoco import gym_envs
from d4rl import infos

init_path = os.path.dirname(os.path.realpath(__file__))


for agent in ['hopper', 'halfcheetah', 'ant', 'walker2d']:
    # mixed
    for dataset in ['medium', 'expert']:
        for version in ['v2']:
            score_env_name = '%s-%s-%s' % (agent, dataset, version)
            for ratio in ["0.01", "0.05", "0.1", "0.5"]:
                env_name = '%s-random-%s-%s-%s' % (agent, dataset, ratio, version)

                register(
                    id='%s-random-%s-%s-%s' % (agent, dataset, ratio, version),
                    entry_point='suboptimal_offline_datasets.custom_envs:get_%s_env' % agent.replace('halfcheetah', 'cheetah').replace('walker2d', 'walker'),
                    max_episode_steps=1000,
                    kwargs={
                        'deprecated': version != 'v2',
                        'ref_min_score': infos.REF_MIN_SCORE[score_env_name],
                        'ref_max_score': infos.REF_MAX_SCORE[score_env_name],
                        'dataset_url': os.path.join(init_path, "custom_datasets", env_name + ".hdf5")
                    }
                )

                register(
                    id='%s-random-%s-partial-%s-%s' % (agent, dataset, ratio, version),
                    entry_point='suboptimal_offline_datasets.custom_envs:get_%s_env' % agent.replace('halfcheetah', 'cheetah').replace('walker2d', 'walker'),
                    max_episode_steps=1000,
                    kwargs={
                        'deprecated': version != 'v2',
                        'ref_min_score': infos.REF_MIN_SCORE[score_env_name],
                        'ref_max_score': infos.REF_MAX_SCORE[score_env_name],
                        'dataset_url': os.path.join(init_path, "partial_mixed_datasets", env_name + ".hdf5")
                    }
                )

    # mixed-varying-n
    for dataset in [
            "medium",
            "expert",
            "medium-expert",
            "full-replay",
            "medium-replay",
        ]:
        for version in ['v2']:
            score_env_name = '%s-%s-%s' % (agent, dataset, version)
            for ratio in ["0.01", "0.05", "0.1", "0.5", "1.0"]:
                if ratio != "1.0" and dataset in ["medium-expert", "full-replay", "medium-replay"]:
                    continue
                for n in [int(1e5), int(5e4), int(2e4), int(1e4), int(5e3)]:
                    if ratio == "1.0":
                        env_name = '%s-%s-%s-%d-%s' % (agent, dataset, ratio, n, version)
                    else:
                        env_name = '%s-random-%s-%s-%d-%s' % (agent, dataset, ratio, n, version)
                    register(
                        id=env_name,
                        entry_point='suboptimal_offline_datasets.custom_envs:get_%s_env' % agent.replace('halfcheetah', 'cheetah').replace('walker2d', 'walker'),
                        max_episode_steps=1000,
                        kwargs={
                            'deprecated': version != 'v2',
                            'ref_min_score': infos.REF_MIN_SCORE[score_env_name],
                            'ref_max_score': infos.REF_MAX_SCORE[score_env_name],
                            'dataset_url': os.path.join(init_path, "custom_datasets", env_name + ".hdf5")
                        }
                    )


    # non-mixed-varying-n
    for dataset in ['medium', 'expert', "medium-expert", "full-replay", "medium-replay"]:
        for version in ['v2']:
            score_env_name = '%s-%s-%s' % (agent, dataset, version)
            ratio = "1.0"
            for n in [int(1e5), int(1e4), int(5e3)]:
                env_name = '%s-random-%s-%s-%d-%s' % (agent, dataset, ratio, n, version)
                register(
                    id=env_name,
                    entry_point='suboptimal_offline_datasets.custom_envs:get_%s_env' % agent.replace('halfcheetah', 'cheetah').replace('walker2d', 'walker'),
                    max_episode_steps=1000,
                    kwargs={
                        'deprecated': version != 'v2',
                        'ref_min_score': infos.REF_MIN_SCORE[score_env_name],
                        'ref_max_score': infos.REF_MAX_SCORE[score_env_name],
                        'dataset_url': os.path.join(init_path, "custom_datasets", env_name + ".hdf5")
                    }
                )

    # reset-stoch
    for dataset in ['medium', 'expert', "random"]:
        for version in ['v2']:
            for mode in ["reset-stoch", "resetfree-stoch", "resetfree-det", "resetfree"]:
                score_env_name = '%s-%s-%s' % (agent, dataset, version)
                register(
                    id=f'{agent}-{dataset}-{mode}-{version}',
                    entry_point='suboptimal_offline_datasets.custom_envs:get_%s_env' % agent.replace('halfcheetah', 'cheetah').replace('walker2d', 'walker'),
                    max_episode_steps=1000,
                    kwargs={
                        'deprecated': version != 'v2',
                        'ref_min_score': infos.REF_MIN_SCORE[score_env_name],
                        'ref_max_score': infos.REF_MAX_SCORE[score_env_name],
                        'dataset_url': os.path.join(init_path, "custom_datasets", f"{agent}-{dataset}-{mode}-v2.hdf5")
                    }
                )


