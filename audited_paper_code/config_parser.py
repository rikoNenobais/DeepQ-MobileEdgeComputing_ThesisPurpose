"""
Configuration Parser for MEC D3QN Offloading Experiments.
Reads and validates .cfg configuration files using Python's standard configparser.
"""

import os
import configparser


DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.cfg')


def get_default_config():
    """Returns default configuration dictionary matching Tang & Wong (Table 1 & Sec 5)."""
    return {
        'ENVIRONMENT': {
            'num_devices': 20,
            'num_edge_nodes': 5,
            'num_time_slots': 100,
            'max_deadline': 10,
            'time_slot_duration': 0.1,
            'f_device': 2.5,
            'f_edge': 41.8,
            'r_tran': 14.0,
            'comp_density': 0.297,
            'penalty_cost': 20.0
        },
        'DRL_AGENT': {
            'learning_rate': 0.001,
            'discount_factor': 0.9,
            'batch_size': 16,
            'replace_threshold': 200,
            'memory_capacity': 500,
            'epsilon_init': 1.0,
            'epsilon_min': 0.01,
            'n_lstm': 20,
            'n_l1': 20
        },
        'WORKLOAD': {
            'pattern': 'sinusoidal',
            'base_arrival_prob': 0.3,
            'min_size': 2.0,
            'max_size': 5.0,
            'size_step': 0.1,
            'period': 50,
            'seed': 42
        },
        'EXPERIMENT': {
            'episodes': 50,
            'enable_epsilon_reset': False,
            'reset_at_episode': 25,
            'eval_episodes': 15,
            'output_dir': 'paper_aligned_code/reports'
        }
    }


def load_config(config_file_path=None):
    """
    Load settings from a .cfg file. If None or not found, falls back to defaults.
    """
    config_dict = get_default_config()

    if config_file_path is None:
        config_file_path = DEFAULT_CONFIG_PATH

    if not os.path.exists(config_file_path):
        print(f"[ConfigParser] Config file '{config_file_path}' not found. Using default parameters.")
        return config_dict

    parser = configparser.ConfigParser()
    parser.read(config_file_path)

    # 1. ENVIRONMENT
    if 'ENVIRONMENT' in parser:
        sec = parser['ENVIRONMENT']
        if 'num_devices' in sec: config_dict['ENVIRONMENT']['num_devices'] = sec.getint('num_devices')
        if 'num_edge_nodes' in sec: config_dict['ENVIRONMENT']['num_edge_nodes'] = sec.getint('num_edge_nodes')
        if 'num_time_slots' in sec: config_dict['ENVIRONMENT']['num_time_slots'] = sec.getint('num_time_slots')
        if 'max_deadline' in sec: config_dict['ENVIRONMENT']['max_deadline'] = sec.getint('max_deadline')
        if 'time_slot_duration' in sec: config_dict['ENVIRONMENT']['time_slot_duration'] = sec.getfloat('time_slot_duration')
        if 'f_device' in sec: config_dict['ENVIRONMENT']['f_device'] = sec.getfloat('f_device')
        if 'f_edge' in sec: config_dict['ENVIRONMENT']['f_edge'] = sec.getfloat('f_edge')
        if 'r_tran' in sec: config_dict['ENVIRONMENT']['r_tran'] = sec.getfloat('r_tran')
        if 'comp_density' in sec: config_dict['ENVIRONMENT']['comp_density'] = sec.getfloat('comp_density')
        if 'penalty_cost' in sec: config_dict['ENVIRONMENT']['penalty_cost'] = sec.getfloat('penalty_cost')

    # 2. DRL_AGENT
    if 'DRL_AGENT' in parser:
        sec = parser['DRL_AGENT']
        if 'learning_rate' in sec: config_dict['DRL_AGENT']['learning_rate'] = sec.getfloat('learning_rate')
        if 'discount_factor' in sec: config_dict['DRL_AGENT']['discount_factor'] = sec.getfloat('discount_factor')
        if 'batch_size' in sec: config_dict['DRL_AGENT']['batch_size'] = sec.getint('batch_size')
        if 'replace_threshold' in sec: config_dict['DRL_AGENT']['replace_threshold'] = sec.getint('replace_threshold')
        if 'memory_capacity' in sec: config_dict['DRL_AGENT']['memory_capacity'] = sec.getint('memory_capacity')
        if 'epsilon_init' in sec: config_dict['DRL_AGENT']['epsilon_init'] = sec.getfloat('epsilon_init')
        if 'epsilon_min' in sec: config_dict['DRL_AGENT']['epsilon_min'] = sec.getfloat('epsilon_min')
        if 'n_lstm' in sec: config_dict['DRL_AGENT']['n_lstm'] = sec.getint('n_lstm')
        if 'n_l1' in sec: config_dict['DRL_AGENT']['n_l1'] = sec.getint('n_l1')

    # 3. WORKLOAD
    if 'WORKLOAD' in parser:
        sec = parser['WORKLOAD']
        if 'pattern' in sec: config_dict['WORKLOAD']['pattern'] = sec.get('pattern').strip().lower()
        if 'base_arrival_prob' in sec: config_dict['WORKLOAD']['base_arrival_prob'] = sec.getfloat('base_arrival_prob')
        if 'min_size' in sec: config_dict['WORKLOAD']['min_size'] = sec.getfloat('min_size')
        if 'max_size' in sec: config_dict['WORKLOAD']['max_size'] = sec.getfloat('max_size')
        if 'size_step' in sec: config_dict['WORKLOAD']['size_step'] = sec.getfloat('size_step')
        if 'period' in sec: config_dict['WORKLOAD']['period'] = sec.getint('period')
        if 'seed' in sec: config_dict['WORKLOAD']['seed'] = sec.getint('seed')

    # 4. EXPERIMENT
    if 'EXPERIMENT' in parser:
        sec = parser['EXPERIMENT']
        if 'episodes' in sec: config_dict['EXPERIMENT']['episodes'] = sec.getint('episodes')
        if 'enable_epsilon_reset' in sec: config_dict['EXPERIMENT']['enable_epsilon_reset'] = sec.getboolean('enable_epsilon_reset')
        if 'reset_at_episode' in sec: config_dict['EXPERIMENT']['reset_at_episode'] = sec.getint('reset_at_episode')
        if 'eval_episodes' in sec: config_dict['EXPERIMENT']['eval_episodes'] = sec.getint('eval_episodes')
        if 'output_dir' in sec: config_dict['EXPERIMENT']['output_dir'] = sec.get('output_dir').strip()

    return config_dict

