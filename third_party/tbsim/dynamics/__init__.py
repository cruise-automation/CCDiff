#  Copyright (c) 2018-present, Cruise LLC
#
#  This source code is licensed under the Apache License, Version 2.0,
#  found in the LICENSE file in the root directory of this source tree.
#  You may not use this file except in compliance with the License.

from typing import Union

from tbsim.dynamics.single_integrator import SingleIntegrator
from tbsim.dynamics.unicycle import Unicycle
from tbsim.dynamics.bicycle import Bicycle
from tbsim.dynamics.double_integrator import DoubleIntegrator
from tbsim.dynamics.base import Dynamics, DynType, forward_dynamics


def get_dynamics_model(dyn_type: Union[str, DynType]):
    if dyn_type in ["Unicycle", DynType.UNICYCLE]:
        return Unicycle
    elif dyn_type == ["SingleIntegrator", DynType.SI]:
        return SingleIntegrator
    elif dyn_type == ["DoubleIntegrator", DynType.DI]:
        return DoubleIntegrator
    else:
        raise NotImplementedError("Dynamics model {} is not implemented".format(dyn_type))
