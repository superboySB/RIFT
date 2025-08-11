#!/usr/bin/env python
# -*- coding: UTF-8 -*-
"""
@File    : __init__.py.py
@Date    : 2024/09/24
"""
from rift.gym_carla.observation.base_observation import EgoBaseObservation, CBVBaseObservation, EgoNormalObservation, CBVNormalObservation, EgoSimpleObservation
from rift.gym_carla.observation.pluto_observation import EgoPlutoObservation, CBVPlutoObservation
from rift.gym_carla.observation.ft_pluto_observation import CBVGRPOPlutoObservation, CBVRIFTPlutoObservation, CBVRSPlutoObservation, CBVRTRPlutoObservation, CBVReinforcePlutoObservation, CBVPPOPlutoObservation, CBVSFTPlutoObservation

OBSERVATION_LIST = {
    'ego_no_obs': EgoBaseObservation,
    'cbv_no_obs': CBVBaseObservation,
    'ego_simple_obs': EgoSimpleObservation,
    'ego_normal_obs': EgoNormalObservation,
    'cbv_normal_obs': CBVNormalObservation,
    'cbv_pluto_obs': CBVPlutoObservation,
    'ego_pluto_obs': EgoPlutoObservation,
    'cbv_rift_pluto_obs': CBVRIFTPlutoObservation,
}
