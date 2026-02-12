from __future__ import annotations

# NOTE: keep imports lightweight.
# `mppi_ball` depends on matplotlib, which may not be installed in every env.

_ball_import_error: Exception | None = None
try:
	from .mppi_ball import MPPI as _BallMPPI
except Exception as e:  # noqa: BLE001
	_BallMPPI = None
	_ball_import_error = e


if _BallMPPI is None:

	class MPPI:  # type: ignore[no-redef]
		def __init__(self, *args, **kwargs):
			raise ModuleNotFoundError(
				"Failed to import algorithms.mppi.MPPI (ball MPPI). "
				"This variant requires extra dependencies (e.g., matplotlib). "
				f"Original error: {_ball_import_error}"
			)

else:
	MPPI = _BallMPPI


try:
	from .mppi_mujoco_arm import MuJoCoArmMPPI
except Exception as e:  # noqa: BLE001

	class MuJoCoArmMPPI:  # type: ignore[no-redef]
		def __init__(self, *args, **kwargs):
			raise ModuleNotFoundError(
				"Failed to import algorithms.mppi.MuJoCoArmMPPI. "
				"Install dependencies like numpy and mujoco. "
				f"Original error: {e}"
			)


__all__ = ["MPPI", "MuJoCoArmMPPI"]
