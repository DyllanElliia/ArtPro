"""Common base for 3D segmentation modules.

``Seg3D_Base`` defines the minimal interface shared by all segmentation
back-ends (PartField, SaMesh, …).

data3d convention
-----------------
A plain ``dict`` whose keys depend on the module:

* Point cloud input::

    {"points":  (N, 3) np.ndarray float32,
     "colors":  (N, 3) np.ndarray float32  -- optional [0,1],
     "normals": (N, 3) np.ndarray float32  -- optional}

* Mesh input::

    {"vertices":  (V, 3) np.ndarray float32,
     "faces":     (F, 3) np.ndarray int32,
     "mesh_path": str                       -- optional, for file-based methods}

Return value of ``seg_3d``
--------------------------
A ``dict`` with at minimum:

* ``"labels"``   : ``(N,)`` or ``(F,)`` ``np.int32`` -- per-element segment IDs.
* ``"features"`` : ``(N, C)`` ``np.float32`` or ``None`` -- intermediate
  features when the back-end produces them (e.g. PartField's 448-D triplane
  features).  Methods that skip feature extraction (e.g. SaMesh) set this
  to ``None``.

seg_fn convention
-----------------
An optional callable passed to ``seg_3d``::

    seg_fn(features: np.ndarray | None, data3d: dict) -> np.ndarray (N,) int

When ``None``, the module uses its built-in default (e.g. k-means for
PartField, or the SAM-lifting pipeline for SaMesh).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, Optional

import numpy as np
import torch


class Seg3D_Base:
  """Abstract base class for 3D segmentation modules.

  Parameters
  ----------
  checkpoint_path : str or None
      Path to the model checkpoint.  Pass ``None`` for methods that do not
      require an explicit checkpoint file (e.g. when the checkpoint is
      specified inside a config file).
  device : str or torch.device
      Target compute device.
  """

  def __init__(
      self,
      checkpoint_path: Optional[str],
      device: str | torch.device,
  ) -> None:
    self.device = torch.device(device)
    self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else None
    if self.checkpoint_path is not None and not self.checkpoint_path.is_file():
      raise FileNotFoundError(self.checkpoint_path)
    self.model = self._build_model()
    if self.model is not None:
      self.model.to(self.device).eval()
      self.model.requires_grad_(False)

  # ── abstract hooks ────────────────────────────────────────────────────

  def _build_model(self) -> Optional[torch.nn.Module]:
    """Construct and return the underlying torch model.

    Return ``None`` if the back-end does not expose a single ``nn.Module``
    (e.g. SaMesh delegates to its own internal pipeline).
    """
    raise NotImplementedError

  @torch.no_grad()
  def extract_features(self, data3d: Dict[str, Any]) -> Optional[np.ndarray]:
    """Run the model forward pass and return per-element features.

    Returns
    -------
    features : (N, C) float32 np.ndarray or None
        Per-element intermediate features.  Return ``None`` for methods
        that produce labels directly without an explicit feature stage.
    """
    raise NotImplementedError

  def seg_3d(
      self,
      data3d: Dict[str, Any],
      seg_fn: Optional[Callable[[Optional[np.ndarray], Dict],
                                np.ndarray]] = None,
  ) -> Dict[str, Any]:
    """Run 3D segmentation.

    Parameters
    ----------
    data3d : dict
        Input 3D data (point cloud or mesh); see module docstring for keys.
    seg_fn : callable, optional
        Custom segmentation function ``(features, data3d) -> labels (N,)
        int``.  When ``None`` the module uses its own default method (e.g.
        k-means for PartField).  For methods that produce labels without
        intermediate features (e.g. SaMesh), ``features`` will be ``None``.

    Returns
    -------
    dict
        ``"labels"``   : ``(N,)`` or ``(F,)`` int32 np.ndarray.
        ``"features"`` : ``(N, C)`` float32 np.ndarray or ``None``.
    """
    raise NotImplementedError
