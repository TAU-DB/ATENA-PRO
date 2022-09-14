import logging
import os
import pickle
from copy import deepcopy
import re

import chainerrl

from arguments import get_args, AlgoName, FilterTermsBinsSizes
from envs import make_env
from models.chinerrl_models.chainerrl_ppo import PPOchianerrl
from models.clipped_gaussian.train_ppo_gym import PPOModel
from models.clipped_gaussian.train_trpo_gym import TRPOModel

import Utilities.Configuration.config as cfg
from gym_atena_pro.envs.atena_pro_env_cont import ATENAPROEnvCont


def reformat_user_input_query(user_uqery):
    lines = [re.sub('and\s*$', '', l).strip() for l in user_uqery.strip().splitlines()]
    return ' and '.join(lines)


def initialize_agent_and_env(is_test=False):
    """
    Read and parse commandline arguments to the args variable.
    Initiate an agent and environment based on the arguments.

    :return: agent, env, args
    """
    args = get_args(is_test)

    # set schema
    cfg.schema = args.schema

    # set dataset nubmer
    cfg.dataset_number = args.dataset_number

    # set number of steps in session
    cfg.MAX_NUM_OF_STEPS = args.episode_length
    # hack to change the default value of the max_steps argument in the __init__ of ATENAPROEnvCont to cfg.MAX_NUM_OF_STEPS
    atena_init_default_args = list(ATENAPROEnvCont.__init__.__defaults__)
    atena_init_default_args[0] = cfg.MAX_NUM_OF_STEPS
    ATENAPROEnvCont.__init__.__defaults__ = tuple(atena_init_default_args)

    # set env settings
    cfg.stack_obs_num = args.stack_obs_num
    cfg.obs_with_step_num = args.obs_with_step_num
    cfg.no_back = args.no_back
    cfg.bins_sizes = args.bins_sizes
    # filter_terms_bins_sizes_helper(FilterTermsBinsSizes(cfg.bins_sizes))
    # paremetric_softmax_idx_action_maps_helper()

    # set reward types to use
    cfg.no_diversity = args.no_diversity
    cfg.no_interestingness = args.no_inter
    cfg.use_humans_reward = args.use_humans_reward
    cfg.humans_reward_interval = args.humans_reward_interval
    cfg.count_data_driven = args.count_data_driven

    # set number of hidden units for gaussian policy
    cfg.obs_with_session_tree = args.obs_with_session_tree
    cfg.obs_stack_history = args.obs_stack_history
    cfg.n_hidden_channels = args.n_hidden_channels
    cfg.n_hidden_layers = args.n_hidden_layers

    # set architecture type
    cfg.arch = args.arch
    cfg.beta = args.beta

    cfg.use_language_reward = not args.no_coherency
    cfg.no_coherency = args.no_coherency

    # optimization settings
    cfg.max_nn_tokens = args.max_nn_tokens
    cfg.cache_dfs_size = args.cache_dfs_size
    cfg.cache_tokenization_size = args.cache_tokenization_size
    cfg.cache_distances_size = args.cache_distances_size

    # set reward coefficients
    cfg.humanity_coeff = args.humanity_coeff
    cfg.diversity_coeff = args.diversity_coeff
    cfg.kl_coeff = args.kl_coeff
    cfg.compaction_coeff = args.compaction_coeff
    cfg.coherency_coeff = args.coherency_coeff

    cfg.compact_observation = args.compact_observation
    cfg.obs_with_session_graph = args.obs_with_session_graph
    cfg.obs_with_session_tree_compact = args.obs_with_session_tree_compact
    cfg.dont_act_deterministically = args.dont_act_deterministically

    cfg.no_distance_reward = args.no_distance_reward
    cfg.no_immediate = args.no_immediate
    cfg.no_buttons = args.no_buttons

    cfg.override_constraints = args.override_constraints
    query_file_path = os.path.join(__file__, os.pardir, os.pardir, os.pardir, 'queries', args.query)
    with open(query_file_path, 'rb') as query_file:
        user_input_query = query_file.read()
    cfg.user_input_query = reformat_user_input_query(user_input_query)

    if args.load or args.demo:
        cfg.is_test = True

    args.outdir = chainerrl.experiments.prepare_output_dir(args, args.outdir)
    cfg.outdir = args.outdir

    # https://stackoverflow.com/questions/13479295/python-using-basicconfig-method-to-log-to-console-and-file
    # logging file path
    log_path = os.path.join(args.outdir, 'training_results.log')
    # set up logging to file
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s',
        filename=log_path,
        datefmt='%H:%M:%S'
    )

    # set up logging to console
    console = logging.StreamHandler()
    console.setLevel(args.logger_level)
    # add the handler to the root logger
    logging.getLogger('').addHandler(console)

    # set logging of the entire episode every LOG_INTERVAL steps
    cfg.log_interval = args.log_interval
    cfg.num_envs = args.num_envs
    ATENAPROEnvCont.LOG_INTERVAL = int(args.log_interval / args.num_envs)

    # TODO (baelo): delete it
    # Set filter term bins
    # filter_terms_bins_sizes_helper(FilterTermsBinsSizes(cfg.bins_sizes))
    # paremetric_softmax_idx_action_maps_helper()

    # Set random seed
    chainerrl.misc.set_random_seed(args.seed, gpus=(args.gpu,))

    # create environment
    env = make_env(args, args.env, args.seed, args.render, args.outdir)

    # set constraints
    if cfg.no_buttons:
        cfg.constraints_dict = dict()
    elif cfg.override_constraints:
        cfg.constraints_dict = env.env_dataset_prop.ENV_CONSTRAINTS_DICT
    else:
        constraints_vec = []
        constraints_str = sorted(set(re.findall('\[(.*?)]', cfg.user_input_query)))
        for constraint_str in constraints_str:
            constraint_vec = env.label_to_action(constraint_str)
            if constraint_vec is not None and constraint_vec not in constraints_vec:
                constraints_vec.append(constraint_vec)
        cfg.constraints_dict = {i: constraint for i, constraint in enumerate(constraints_vec)}

    # choose algorithm
    args.algo = AlgoName(args.algo)
    if args.algo is AlgoName.CAPG_PPO:  # capg
        model = PPOModel(env,
                         args.gpu,
                         args.n_hidden_channels,
                         args.adam_lr,
                         args.ppo_update_interval,
                         args.outdir,
                         args.load,
                         args.use_clipped_gaussian)
    elif args.algo is AlgoName.CAPG_TRPO:  # capg
        model = TRPOModel(env,
                          args.gpu,
                          args.n_hidden_channels,
                          args.trpo_update_interval,
                          args.outdir,
                          args.load,
                          args.use_clipped_gaussian)
    elif args.algo == AlgoName.CHAINERRL_PPO:
        model = PPOchianerrl(args, env)
    else:
        raise NotImplementedError

    agent = model.agent

    return agent, env, args


def setup_logger(name, log_file, level=logging.INFO):
    """Function setup as many loggers as you want"""
    # https://stackoverflow.com/questions/11232230/logging-to-two-files-with-different-settings

    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')

    handler = logging.FileHandler(log_file)
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    return logger
