#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@File    : __init__.py
@Date    : 2024/8/15
"""
# collect policy models from scenarios
from rift.cbv.planning.dummy_policy import DummyPolicy
from rift.cbv.planning.fine_tuner.rlft.rift_pluto.rift_pluto import RIFTPluto
from rift.cbv.planning.pluto.pluto import PLUTO


CBV_POLICY_LIST = {
    'standard': DummyPolicy,
    'pluto': PLUTO,
    'rift_pluto': RIFTPluto,
}
