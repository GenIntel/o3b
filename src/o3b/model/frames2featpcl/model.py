import logging
logger = logging.getLogger(__name__)
from o3b.model.model import OD3D_Model, register_model
from o3b.data.datatypes.frame import FrameBatch
from o3b.model.frames2featpcl.frames2featpcl import frames2featpcl

@register_model("Frames2FeatPCL")
class Frames2FeatPCL(OD3D_Model):
    def __init__(
        self,
        append_rays3d=True,
        normalize_feats=True,
        filter_median=False,
        pts_count=1024,
        featmap_upsample_rate=2,
        normalize_pts3d=False,
        augment_rot = True, # : False # True 
        augment_scale_dev =0.2, # : 0. # 0.2
        augment_transl_dev = 0.1, # : 0. # 0.1
        scale_quantile=0.9, # : 0.9
        pts3d_opengl=False, # unproject into OpenGL camera space; see frames2featpcl
    ):
        super().__init__()
        self.append_rays3d = append_rays3d
        self.normalize_feats = normalize_feats
        self.filter_median = filter_median
        self.pts_count = pts_count
        self.featmap_upsample_rate = featmap_upsample_rate
        self.normalize_pts3d = normalize_pts3d
        self.augment_rot = augment_rot
        self.augment_scale_dev = augment_scale_dev
        self.augment_transl_dev = augment_transl_dev
        self.scale_quantile = scale_quantile
        self.pts3d_opengl = pts3d_opengl
        
    def forward(self, frames_gt: FrameBatch, frames_pred: FrameBatch = None):
        frames_gt, frames_pred = frames2featpcl(frames_gt, frames_pred, 
                                                pts_count=self.pts_count, 
                                                append_rays3d=self.append_rays3d,
                                                normalize_feats=self.normalize_feats,
                                                filter_median=self.filter_median,
                                                featmap_upsample_rate=self.featmap_upsample_rate,
                                                normalize_pts3d=self.normalize_pts3d, 
                                                training=self.training, 
                                                augment_rot=self.augment_rot,
                                                augment_scale_dev=self.augment_scale_dev,
                                                augment_transl_dev=self.augment_transl_dev,
                                                scale_quantile=self.scale_quantile,
                                                opengl=self.pts3d_opengl)
        return frames_gt, frames_pred