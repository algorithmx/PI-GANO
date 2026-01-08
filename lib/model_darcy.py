import torch
import torch.nn as nn

from lib.model_base import (
    BaseDCON as _UnifiedBaseDCON,
    BaseGANO as _UnifiedBaseGANO,
    BaseNeuralOperator,
    DCONBlendMixin,
    DomainGeometry,
    DomainGeometryOtherEmbedding,
    build_mlp,
    build_sequential_layers,
)

'''
Physics-informed neural operator models for 2D Darcy flow problem on variable domain geometries.

Refactored with class hierarchy to reduce code duplication following PI-DCON/Main/models.py patterns.

- Darcy: predicts a scalar field u(x, y)
- Uses masked max-pooling for variable-size boundary inputs
- Supports various geometry-aware coupling strategies
'''

class BaseDCON(_UnifiedBaseDCON):
    """Compatibility wrapper for Darcy: scalar field_dim=1, par_dim=3."""

    def __init__(self, config):
        super().__init__(field_dim=1, config=config, par_dim=3)

    def _zip(self, xy, par, par_flag, enc=None):
        # Keep the historical Darcy signature (no explicit axis)
        return super()._zip(xy, par, axis=0, enc=enc, par_flag=par_flag)


class BaseGANO(_UnifiedBaseGANO):
    """Compatibility wrapper for Darcy GANO: field_dim=1, par_dim=3."""

    def __init__(self, config):
        super().__init__(field_dim=1, config=config, par_dim=3)

    def _zip(self, xy_global, par, par_flag, enc=None):
        # Keep Darcy signature (no explicit axis)
        return super()._zip(xy_global, par, axis=0, enc=enc, par_flag=par_flag)


# ============================================================================
# Baseline Models
# ============================================================================

''' ------------------------- PI-DCON -------------------------- '''

class PI_DCON(BaseDCON):
    """Physics-informed DCON baseline for Darcy flow.
    
    Uses masked max-pooling over boundary conditions and gated modulation
    in trunk network. Does not use geometry encoder.
    """

    def __init__(self, config):
        super().__init__(config)
        # All layers inherited from BaseDCON

    def forward(self, x_coor, y_coor, par, par_flag, shape_coor, shape_flag):
        '''
        model input:
            x_coor (B, M): x-axis coordinates of the collocation points
            y_coor (B, M): y-axis coordinates of the collocation points
            par (B, M', 3): boundary coordinates and function values (x, y, u)
            par_flag (B, M'): valid flags for boundary points
            shape_coor: (B, M'', 2) - not used in DCON
            shape_flag: (B, M'') - not used in DCON

        model output:
            u (B, M): scalar field values over the domain
        '''
        # Concat coordinates
        xy = torch.cat((x_coor.unsqueeze(-1), y_coor.unsqueeze(-1)), -1)

        # Predict using unified _zip method
        u = self._zip(xy, par, par_flag)

        return u


''' ------------------------- PI-PN -------------------------- '''

class PI_PN(BaseNeuralOperator):
    """Physics-informed PointNet baseline"""

    def __init__(self, config):
        super().__init__(config)

        # encoder network
        self.encoder = nn.Sequential(*build_mlp(3, self.fc_dim, self.n_layer, self.fc_dim))

        # decoder network
        self.decoder = nn.Sequential(*build_mlp(2 * self.fc_dim, self.fc_dim, self.n_layer, 1))


    def forward(self, x_coor, y_coor, par, par_flag):
        '''
        par: (B, M', 3)
        par_flag: (B, M')
        x_coor: (B, M)
        y_coor: (B, M)

        return u: (B, M)
        '''

        # get the hidden embeddings
        xyf = torch.cat((x_coor.unsqueeze(-1), y_coor.unsqueeze(-1), par.unsqueeze(-1)), -1)
        enc = self.encoder(xyf)    # (B, M, F)

        # global feature embeddings
        enc_masked = enc * par_flag.unsqueeze(-1)    # (B, M, F)
        F_G = torch.amax(enc_masked, 1, keepdim=True)    # (B, 1, F)
        enc = torch.cat((enc, F_G.repeat(1,enc.shape[1],1)), -1)    # (B, M, 2F)

        # decode
        pred = self.decoder(enc)    # (B, M, 1)
        u = pred.squeeze(-1)    # (B, M)

        return u


class PI_PN_only_geo(BaseNeuralOperator):
    """Physics-informed PointNet for fixed PDE parameters (geometry-only)"""

    def __init__(self, config):
        super().__init__(config)

        # encoder network
        self.encoder = nn.Sequential(*build_mlp(2, self.fc_dim, self.n_layer, self.fc_dim))

        # decoder network
        self.decoder = nn.Sequential(*build_mlp(2 * self.fc_dim, self.fc_dim, self.n_layer, 1))


    def forward(self, x_coor, y_coor, par, par_flag):
        '''
        par: (B, M', 3)
        par_flag: (B, M')
        x_coor: (B, M)
        y_coor: (B, M)

        return u: (B, M)
        '''

        # get the hidden embeddings
        xyf = torch.cat((x_coor.unsqueeze(-1), y_coor.unsqueeze(-1)), -1)
        enc = self.encoder(xyf)    # (B, M, F)

        # global feature embeddings
        enc_masked = enc * par_flag.unsqueeze(-1)    # (B, M, F)
        F_G = torch.amax(enc_masked, 1, keepdim=True)    # (B, 1, F)
        enc = torch.cat((enc, F_G.repeat(1,enc.shape[1],1)), -1)    # (B, M, 2F)

        # decode
        pred = self.decoder(enc)    # (B, M, 1)
        u = pred.squeeze(-1)    # (B, M)

        return u


# ============================================================================
# PI-GANO Main Architecture
# ============================================================================

''' ------------------------- PI-GANO (concatenation coupling) -------------------------- '''

class PI_GANO(BaseGANO):
    """
    Physics-Informed Geometry-Aware Neural Operator with concatenation coupling.
    Main GANO architecture proposed in the paper.
    
    Uses domain geometry encoder + coordinate lifting + concatenation coupling.
    """

    def __init__(self, config):
        super().__init__(config)

        # Branch network (2*fc_dim for concatenated features)
        self.branch = nn.Sequential(*build_mlp(3, 2*self.fc_dim, self.n_layer, 2*self.fc_dim))

        # Coordinate lifting layer
        self.xy_lift = nn.Linear(2, self.fc_dim)

        # Trunk network (operates on 2*fc_dim concatenated features)
        self.FC = nn.ModuleList([
            nn.ModuleList(build_sequential_layers(2*self.fc_dim, 2*self.fc_dim, 3)),
        ])


    def forward(self, x_coor, y_coor, par, par_flag, shape_coor, shape_flag):
        '''
        model input:
            x_coor (B, M): x-axis coordinates of the collocation points
            y_coor (B, M): y-axis coordinates of the collocation points
            par (B, M', 3): boundary coordinates and function values (x, y, u)
            par_flag (B, M'): valid flags for boundary points
            shape_coor (B, M'', 2): domain boundary shape coordinates
            shape_flag (B, M''): valid flags for shape points

        model output:
            u (B, M): scalar field values over the domain
        '''
        # Forward to get the domain embedding
        Domain_enc = self._encode_geometry(shape_coor, shape_flag)    # (B,1,F)

        # Concat coordinates
        xy = torch.cat((x_coor.unsqueeze(-1), y_coor.unsqueeze(-1)), -1)

        # Lift the dimension of coordinate embedding
        xy_local = self.xy_lift(xy)   # (B,M,F)

        # Combine with global embedding (concatenation)
        xy_global = self._combine_embeddings(xy_local, Domain_enc, coupling='concat')    # (B,M,2F)

        # Predict using unified _zip method
        u = self._zip(xy_global, par, par_flag)

        return u


''' ------------------------- PI-GANO variants with different coupling -------------------------- '''

class PI_GANO_add(BaseGANO):
    """PI-GANO with addition coupling for feature fusion.
    
    Uses element-wise addition to combine local coordinates with global geometry.
    """

    def __init__(self, config):
        super().__init__(config)

        # Use alternative geometry encoder with 2*fc_dim output
        self.DG = DomainGeometryOtherEmbedding(config)

        # Branch network
        self.branch = nn.Sequential(*build_mlp(3, 2*self.fc_dim, self.n_layer, 2*self.fc_dim))

        # Coordinate lifting layer (to 2F for addition)
        self.xy_lift = nn.Linear(2, 2*self.fc_dim)

        # Trunk network
        self.FC = nn.ModuleList([
            nn.ModuleList(build_sequential_layers(2*self.fc_dim, 2*self.fc_dim, 3)),
        ])


    def forward(self, x_coor, y_coor, par, par_flag, shape_coor, shape_flag):
        '''
        model input:
            x_coor (B, M): x-axis coordinates of the collocation points
            y_coor (B, M): y-axis coordinates of the collocation points
            par (B, M', 3): boundary coordinates and function values (x, y, u)
            par_flag (B, M'): valid flags for boundary points
            shape_coor (B, M'', 2): domain boundary shape coordinates
            shape_flag (B, M''): valid flags for shape points

        model output:
            u (B, M): scalar field values over the domain
        '''
        # Forward to get the domain embedding
        Domain_enc = self._encode_geometry(shape_coor, shape_flag)    # (B,1,2F)

        # Concat coordinates
        xy = torch.cat((x_coor.unsqueeze(-1), y_coor.unsqueeze(-1)), -1)

        # Lift the dimension of coordinate embedding
        xy_local = self.xy_lift(xy)   # (B,M,2F)

        # Combine with global embedding (addition)
        xy_global = self._combine_embeddings(xy_local, Domain_enc, coupling='add')    # (B,M,2F)

        # Predict using unified _zip method
        u = self._zip(xy_global, par, par_flag)

        return u


class PI_GANO_mul(BaseGANO):
    """PI-GANO with element-wise multiplication coupling for feature fusion.
    
    Uses element-wise multiplication to combine local coordinates with global geometry.
    """

    def __init__(self, config):
        super().__init__(config)

        # Use alternative geometry encoder with 2*fc_dim output
        self.DG = DomainGeometryOtherEmbedding(config)

        # Branch network
        self.branch = nn.Sequential(*build_mlp(3, 2*self.fc_dim, self.n_layer, 2*self.fc_dim))

        # Coordinate lifting layer (to 2F for multiplication)
        self.xy_lift = nn.Linear(2, 2*self.fc_dim)

        # Trunk network
        self.FC = nn.ModuleList([
            nn.ModuleList(build_sequential_layers(2*self.fc_dim, 2*self.fc_dim, 3)),
        ])


    def forward(self, x_coor, y_coor, par, par_flag, shape_coor, shape_flag):
        '''
        model input:
            x_coor (B, M): x-axis coordinates of the collocation points
            y_coor (B, M): y-axis coordinates of the collocation points
            par (B, M', 3): boundary coordinates and function values (x, y, u)
            par_flag (B, M'): valid flags for boundary points
            shape_coor (B, M'', 2): domain boundary shape coordinates
            shape_flag (B, M''): valid flags for shape points

        model output:
            u (B, M): scalar field values over the domain
        '''
        # Forward to get the domain embedding
        Domain_enc = self._encode_geometry(shape_coor, shape_flag)    # (B,1,2F)

        # Concat coordinates
        xy = torch.cat((x_coor.unsqueeze(-1), y_coor.unsqueeze(-1)), -1)

        # Lift the dimension of coordinate embedding
        xy_local = self.xy_lift(xy)   # (B,M,2F)

        # Combine with global embedding (multiplication)
        xy_global = self._combine_embeddings(xy_local, Domain_enc, coupling='mul')    # (B,M,2F)

        # Predict using unified _zip method
        u = self._zip(xy_global, par, par_flag)

        return u


class PI_GANO_geo(DCONBlendMixin, BaseNeuralOperator):
    """
    PI-GANO variant using high-level geometric features instead of point clouds.
    Uses parametric geometry representation (e.g., shape parameters) as input.
    """

    def __init__(self, config, geo_feature_dim):
        super().__init__(config)

        # Geometry encoder from high-level features
        self.geo_encoder = nn.Sequential(*build_mlp(geo_feature_dim, self.fc_dim, self.n_layer, self.fc_dim))

        # Branch network
        self.branch = nn.Sequential(*build_mlp(3, 2*self.fc_dim, self.n_layer, 2*self.fc_dim))

        # Coordinate lifting layer
        self.xy_lift = nn.Linear(2, self.fc_dim)

        # Trunk network
        self.FC = nn.ModuleList([
            nn.ModuleList(build_sequential_layers(2*self.fc_dim, 2*self.fc_dim, 3)),
        ])

        # Activation
        self.act = nn.Tanh()

    def _encode_par(self, par, par_flag):
        """Compute the parameter encoding from branch network."""
        enc = self.branch(par)
        enc_masked = enc * par_flag.unsqueeze(-1)
        enc = torch.amax(enc_masked, 1, keepdim=True)
        return enc

    def forward(self, x_coor, y_coor, par, par_flag, geo_feature):
        '''
        model input:
            x_coor (B, M): x-axis coordinates of the collocation points
            y_coor (B, M): y-axis coordinates of the collocation points
            par (B, M', 3): boundary coordinates and function values (x, y, u)
            par_flag (B, M'): valid flags for boundary points
            geo_feature (B, geo_feature_dim): high-level geometric features

        model output:
            u (B, M): scalar field values over the domain
        '''
        # Extract number of points
        B, mD = x_coor.shape

        # Forward to get the domain embedding from high-level features
        Domain_enc = self.geo_encoder(geo_feature).unsqueeze(1)    # (B,1,F)

        # Concat coordinates
        xy = torch.cat((x_coor.unsqueeze(-1), y_coor.unsqueeze(-1)), -1)

        # Lift the dimension of coordinate embedding
        xy_local = self.xy_lift(xy)   # (B,M,F)

        # Combine with global embedding (concatenation)
        xy_global = torch.cat((xy_local, Domain_enc.repeat(1,mD,1)), -1)    # (B,M,2F)

        # Get the kernels
        enc = self._encode_par(par, par_flag)

        # Predict u using mixin method
        u = self._predict_head(xy_global, enc, self.FC[0])

        return u


''' ------------------------- New model template -------------------------- '''

class New_model_darcy(nn.Module):

    def __init__(self, config):
        super().__init__()

    def forward(self, x_coor, y_coor, par, par_flag, shape_coor, shape_flag):
        '''
        par: (B, M', 3)
        par_flag: (B, M')
        x_coor: (B, M)
        y_coor: (B, M)

        return u: (B, M)
        '''

        return None
