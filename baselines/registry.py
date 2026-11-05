"""Registered algorithm and suite entry points."""

ALGO_REGISTRY: dict[tuple[str, str], tuple[str, str, str]] = {
    ("ippo_ff", "mpe"): ("baselines/IPPO/ippo_ff_mpe.py", "", ""),
    ("ippo_rnn", "mpe"): ("baselines/IPPO/ippo_rnn_mpe.py", "", ""),
    ("i2c", "mpe"): ("baselines/I2C/i2c_mpe.py", "i2c_mpe", "alg."),
    ("mappo_rnn", "mpe"): ("baselines/MAPPO/mappo_rnn_mpe.py", "", ""),
    ("qmix_rnn", "mpe"): ("baselines/QLearning/qmix_rnn.py", "ql_rnn_mpe", "alg."),
    ("vdn_rnn", "mpe"): ("baselines/QLearning/vdn_rnn.py", "ql_rnn_mpe", "alg."),
    ("vdn_ff", "mpe"): ("baselines/QLearning/vdn_ff.py", "vdn_ff_mpe", "alg."),
    ("ippo_rnn", "smax"): ("baselines/IPPO/ippo_rnn_smax.py", "", ""),
    ("mappo_rnn", "smax"): ("baselines/MAPPO/mappo_rnn_smax.py", "", ""),
    ("qmix_rnn", "smax"): ("baselines/QLearning/qmix_rnn.py", "ql_rnn_smax", "alg."),
    ("rode", "smax"): ("baselines/RODE/rode.py", "rode_smax", "alg."),
    ("ndq", "smax"): ("baselines/NDQ/ndq_smax.py", "ndq_smax", "alg."),
    ("ndo", "smax"): ("baselines/NDQ/ndq_smax.py", "ndq_smax", "alg."),
    ("vdn_rnn", "smax"): ("baselines/QLearning/vdn_rnn.py", "ql_rnn_smax", "alg."),
    ("facmac_smac", "smax"): (
        "baselines/FACMAC/facmac.py",
        "facmac_smac",
        "alg.",
    ),
    ("facmac_mamujoco", "mamujoco"): (
        "baselines/FACMAC/facmac.py",
        "facmac_mamujoco",
        "alg.",
    ),
    ("ippo_ff", "overcooked"): ("baselines/IPPO/ippo_ff_overcooked.py", "", ""),
    (
        "vdn_cnn",
        "overcooked",
    ): ("baselines/QLearning/vdn_cnn_overcooked.py", "ql_cnn_overcooked", "alg."),
    ("ippo_ff", "hanabi"): ("baselines/IPPO/ippo_ff_hanabi.py", "", ""),
    ("mappo_ff", "hanabi"): ("baselines/MAPPO/mappo_ff_hanabi.py", "", ""),
}

# FACMAC uses its native Sacred/PyTorch runner.  The tuple contains the
# algorithm config, environment config, and the environment argument that
# selects a scenario or map.
FACMAC_REGISTRY: dict[tuple[str, str], tuple[str, str, str]] = {
    ("facmac_pp", "particle"): ("facmac_pp", "particle", "scenario_name"),
    ("maddpg_pp", "particle"): ("maddpg_pp", "particle", "scenario_name"),
    ("maddpg_smac", "smax"): ("maddpg_smac", "sc2", "map_name"),
    ("maddpg_mamujoco", "mamujoco"): (
        "maddpg_mamujoco",
        "mujoco_multi",
        "scenario_name",
    ),
    ("comix_pp", "particle"): ("comix_pp", "particle", "scenario_name"),
    ("covdn_pp", "particle"): ("covdn_pp", "particle", "scenario_name"),
    ("iddpg_pp", "particle"): ("iddpg_pp", "particle", "scenario_name"),
}
