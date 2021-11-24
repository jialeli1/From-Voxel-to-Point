from .partA2_head import PartA2FCHead
from .pointrcnn_head import PointRCNNHead
from .pvrcnn_head import PVRCNNHead
from .roi_head_template import RoIHeadTemplate
from .pointrcnniou_head import PointRCNNIoUHead
from .voxelrcnn_head import VoxelRCNNHead


from .iouguided_roi_head import IoUGuidedRoIHead

__all__ = {
    'RoIHeadTemplate': RoIHeadTemplate,
    'PartA2FCHead': PartA2FCHead,
    'PVRCNNHead': PVRCNNHead,
    'PointRCNNHead': PointRCNNHead,
    'PointRCNNIoUHead': PointRCNNIoUHead,
    'VoxelRCNNHead': VoxelRCNNHead,

    'IoUGuidedRoIHead': IoUGuidedRoIHead,
}
