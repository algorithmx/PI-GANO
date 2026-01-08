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
Physics-informed neural operator models for 2D plate stress problem on variable domain geometries.

Refactored with class hierarchy to reduce code duplication following PI-DCON/Main/models.py patterns.

- Plate: predicts a 2-component vector field (u(x, y), v(x, y))
- Uses masked max-pooling for variable-size boundary inputs
- Supports various geometry-aware coupling strategies
'''

class BaseDCON_plate(_UnifiedBaseDCON):
    """Compatibility wrapper for Plate: vector field_dim=2, par_dim=4."""

    def __init__(self, config):
        super().__init__(field_dim=2, config=config, par_dim=4)

    def _zip(self, xy, par, par_flag, axis, enc=None):
        return super()._zip(xy, par, axis=axis, enc=enc, par_flag=par_flag)


class BaseGANO(_UnifiedBaseGANO):
    """Compatibility wrapper for Plate GANO: field_dim=2, par_dim=4."""

    def __init__(self, config):
        super().__init__(field_dim=2, config=config, par_dim=4)

    def _zip(self, xy_global, par, par_flag, axis, enc=None):
        return super()._zip(xy_global, par, axis=axis, enc=enc, par_flag=par_flag)


# ============================================================================
# Baseline Models
# ============================================================================

''' ------------------------- PI-DCON -------------------------- '''

class PI_DCON_plate(BaseDCON_plate):
    """Physics-informed DCON baseline for plate stress problem.
    
    Uses masked max-pooling over boundary conditions and gated modulation
    in trunk network. Does not use geometry encoder.
    Outputs 2-component vector field (u, v).
    """

    def __init__(self, config):
        super().__init__(config)
        # All layers inherited from BaseDCON_plate

    def forward(self, x_coor, y_coor, par, par_flag, shape_coor, shape_flag):
        '''
        model input:
            x_coor (B, M): x-axis coordinates of the collocation points
            y_coor (B, M): y-axis coordinates of the collocation points
            par (B, M', 4): boundary coordinates and function values (x, y, u, v)
            par_flag (B, M'): valid flags for boundary points
            shape_coor: (B, M'', 2) - not used in DCON
            shape_flag: (B, M'') - not used in DCON

        model output:
            u (B, M): x-component of the vector field
            v (B, M): y-component of the vector field
        '''
        # Get the kernel (compute once, shared by both components)
        enc = self._encode_par(par, par_flag)

        # Concat coordinates
        xy = torch.cat((x_coor.unsqueeze(-1), y_coor.unsqueeze(-1)), -1)

        # Predict u and v using shared encoding
        u = self._zip(xy, par, par_flag, 0, enc)
        v = self._zip(xy, par, par_flag, 1, enc)

        return u, v


''' ------------------------- PI-PN -------------------------- '''

class PI_PN_plate(BaseNeuralOperator):
    """Physics-informed PointNet baseline for plate stress"""

    def __init__(self, config):
        super().__init__(config)

        # encoder network
        self.encoder = nn.Sequential(*build_mlp(4, self.fc_dim, self.n_layer, self.fc_dim))

        # decoder network (outputs 2 components)
        self.decoder = nn.Sequential(*build_mlp(2 * self.fc_dim, self.fc_dim, self.n_layer, 2))


    def forward(self, x_coor, y_coor, u_input, v_input, flag):
        '''
        x_coor: (B, M)
        y_coor: (B, M)
        u_input: (B, M)
        v_input: (B, M)
        flag: (B, M)

        return u, v: (B, M), (B, M)
        '''

        # construct inputs
        xyf = torch.cat((x_coor.unsqueeze(-1), y_coor.unsqueeze(-1), u_input.unsqueeze(-1), v_input.unsqueeze(-1)), -1)

        # get the hidden embeddings
        enc = self.encoder(xyf)    # (B, M, F)
        enc_masked = enc * flag.unsqueeze(-1)    # (B, M, F)
        F_G = torch.amax(enc_masked, 1, keepdim=True)    # (B, 1, F)

        # combine
        enc = torch.cat((enc, F_G.repeat(1,enc.shape[1],1)), -1)    # (B, M, 2F)

        # decode
        pred = self.decoder(enc)    # (B, M, 2)
        u = pred[:,:,0]
        v = pred[:,:,1]

        return u, v


class PI_PN_only_geo_plate(BaseNeuralOperator):
    """Physics-informed PointNet for fixed PDE parameters (geometry-only)"""

    def __init__(self, config):
        super().__init__(config)

        # encoder network
        self.encoder = nn.Sequential(*build_mlp(2, self.fc_dim, self.n_layer, self.fc_dim))

        # decoder network (outputs 2 components)
        self.decoder = nn.Sequential(*build_mlp(2 * self.fc_dim, self.fc_dim, self.n_layer, 2))


    def forward(self, x_coor, y_coor, flag):
        '''
        x_coor: (B, M)
        y_coor: (B, M)
        flag: (B, M)

        return u, v: (B, M), (B, M)
        '''

        # construct inputs
        xyf = torch.cat((x_coor.unsqueeze(-1), y_coor.unsqueeze(-1)), -1)

        # get the hidden embeddings
        enc = self.encoder(xyf)    # (B, M, F)
        enc_masked = enc * flag.unsqueeze(-1)    # (B, M, F)
        F_G = torch.amax(enc_masked, 1, keepdim=True)    # (B, 1, F)

        # combine
        enc = torch.cat((enc, F_G.repeat(1,enc.shape[1],1)), -1)    # (B, M, 2F)

        # decode
        pred = self.decoder(enc)    # (B, M, 2)
        u = pred[:,:,0]
        v = pred[:,:,1]

        return u, v


# ============================================================================
# PI-GANO Main Architecture
# ============================================================================

''' ------------------------- PI-GANO (concatenation coupling) -------------------------- '''

class PI_GANO_plate(BaseGANO):
    """
    Physics-Informed Geometry-Aware Neural Operator with concatenation coupling.
    Main GANO architecture for plate stress problem (2 components).
    
    Uses domain geometry encoder + coordinate lifting + concatenation coupling.
    """

    def __init__(self, config):
        super().__init__(config)

        # Branch network (2*fc_dim for concatenated features)
        self.branch = nn.Sequential(*build_mlp(4, 2*self.fc_dim, self.n_layer, 2*self.fc_dim))

        # Coordinate lifting layers (separate for u and v)
        self.xy_lift1 = nn.Linear(2, self.fc_dim)
        self.xy_lift2 = nn.Linear(2, self.fc_dim)

        # Trunk networks for u and v components (4 layers each)
        self.FC = nn.ModuleList([
            nn.ModuleList(build_sequential_layers(2*self.fc_dim, 2*self.fc_dim, 4)),  # for u
            nn.ModuleList(build_sequential_layers(2*self.fc_dim, 2*self.fc_dim, 4)),  # for v
        ])


    def forward(self, x_coor, y_coor, par, par_flag, shape_coor, shape_flag):
        '''
        model input:
            x_coor (B, M): x-axis coordinates of the collocation points
            y_coor (B, M): y-axis coordinates of the collocation points
            par (B, M', 4): boundary coordinates and function values (x, y, u, v)
            par_flag (B, M'): valid flags for boundary points
            shape_coor (B, M'', 2): domain boundary shape coordinates
            shape_flag (B, M''): valid flags for shape points

        model output:
            u (B, M): x-component of the vector field
            v (B, M): y-component of the vector field
        '''
        # Forward to get the domain embedding
        Domain_enc = self._encode_geometry(shape_coor, shape_flag)    # (B,1,F)

        # Concat coordinates
        xy = torch.cat((x_coor.unsqueeze(-1), y_coor.unsqueeze(-1)), -1)

        # Lift the dimension of coordinate embedding (separate for u and v)
        xy_local_u = self.xy_lift1(xy)   # (B,M,F)
        xy_local_v = self.xy_lift2(xy)   # (B,M,F)

        # Combine with global embedding (concatenation)
        xy_global_u = self._combine_embeddings(xy_local_u, Domain_enc, coupling='concat')    # (B,M,2F)
        xy_global_v = self._combine_embeddings(xy_local_v, Domain_enc, coupling='concat')    # (B,M,2F)

        # Get the kernels (compute once, shared by both components)
        enc = self._encode_par(par, par_flag)

        # Predict u and v using unified _zip method with shared encoding
        u = self._zip(xy_global_u, par, par_flag, 0, enc)
        v = self._zip(xy_global_v, par, par_flag, 1, enc)

        return u, v


''' ------------------------- PI-GANO variants with different coupling -------------------------- '''

class PI_GANO_add_plate(BaseGANO):
    """PI-GANO with addition coupling for feature fusion (plate version).
    
    Uses element-wise addition to combine local coordinates with global geometry.
    """

    def __init__(self, config):
        super().__init__(config)

        # Use alternative geometry encoder with 2*fc_dim output
        self.DG = DomainGeometryOtherEmbedding(config)

        # Branch network
        self.branch = nn.Sequential(*build_mlp(4, 2*self.fc_dim, self.n_layer, 2*self.fc_dim))

        # Coordinate lifting layers (separate for u and v, to 2F for addition)
        self.xy_lift1 = nn.Linear(2, 2*self.fc_dim)
        self.xy_lift2 = nn.Linear(2, 2*self.fc_dim)

        # Trunk networks for u and v components (4 layers each)
        self.FC = nn.ModuleList([
            nn.ModuleList(build_sequential_layers(2*self.fc_dim, 2*self.fc_dim, 4)),  # for u
            nn.ModuleList(build_sequential_layers(2*self.fc_dim, 2*self.fc_dim, 4)),  # for v
        ])


    def forward(self, x_coor, y_coor, par, par_flag, shape_coor, shape_flag):
        '''
        model input:
            x_coor (B, M): x-axis coordinates of the collocation points
            y_coor (B, M): y-axis coordinates of the collocation points
            par (B, M', 4): boundary coordinates and function values (x, y, u, v)
            par_flag (B, M'): valid flags for boundary points
            shape_coor (B, M'', 2): domain boundary shape coordinates
            shape_flag (B, M''): valid flags for shape points

        model output:
            u (B, M): x-component of the vector field
            v (B, M): y-component of the vector field
        '''
        # Forward to get the domain embedding
        Domain_enc = self._encode_geometry(shape_coor, shape_flag)    # (B,1,2F)

        # Concat coordinates
        xy = torch.cat((x_coor.unsqueeze(-1), y_coor.unsqueeze(-1)), -1)

        # Lift the dimension of coordinate embedding
        xy_local_u = self.xy_lift1(xy)   # (B,M,2F)
        xy_local_v = self.xy_lift2(xy)   # (B,M,2F)

        # Combine with global embedding (addition)
        xy_global_u = self._combine_embeddings(xy_local_u, Domain_enc, coupling='add')    # (B,M,2F)
        xy_global_v = self._combine_embeddings(xy_local_v, Domain_enc, coupling='add')    # (B,M,2F)

        # Get the kernels (compute once, shared by both components)
        enc = self._encode_par(par, par_flag)

        # Predict u and v using unified _zip method
        u = self._zip(xy_global_u, par, par_flag, 0, enc)
        v = self._zip(xy_global_v, par, par_flag, 1, enc)

        return u, v


class PI_GANO_mul_plate(BaseGANO):
    """PI-GANO with element-wise multiplication coupling for feature fusion (plate version).
    
    Uses element-wise multiplication to combine local coordinates with global geometry.
    """

    def __init__(self, config):
        super().__init__(config)

        # Use alternative geometry encoder with 2*fc_dim output
        self.DG = DomainGeometryOtherEmbedding(config)

        # Branch network
        self.branch = nn.Sequential(*build_mlp(4, 2*self.fc_dim, self.n_layer, 2*self.fc_dim))

        # Coordinate lifting layers (separate for u and v, to 2F for multiplication)
        self.xy_lift1 = nn.Linear(2, 2*self.fc_dim)
        self.xy_lift2 = nn.Linear(2, 2*self.fc_dim)

        # Trunk networks for u and v components (4 layers each)
        self.FC = nn.ModuleList([
            nn.ModuleList(build_sequential_layers(2*self.fc_dim, 2*self.fc_dim, 4)),  # for u
            nn.ModuleList(build_sequential_layers(2*self.fc_dim, 2*self.fc_dim, 4)),  # for v
        ])


    def forward(self, x_coor, y_coor, par, par_flag, shape_coor, shape_flag):
        '''
        model input:
            x_coor (B, M): x-axis coordinates of the collocation points
            y_coor (B, M): y-axis coordinates of the collocation points
            par (B, M', 4): boundary coordinates and function values (x, y, u, v)
            par_flag (B, M'): valid flags for boundary points
            shape_coor (B, M'', 2): domain boundary shape coordinates
            shape_flag (B, M''): valid flags for shape points

        model output:
            u (B, M): x-component of the vector field
            v (B, M): y-component of the vector field
        '''
        # Forward to get the domain embedding
        Domain_enc = self._encode_geometry(shape_coor, shape_flag)    # (B,1,2F)

        # Concat coordinates
        xy = torch.cat((x_coor.unsqueeze(-1), y_coor.unsqueeze(-1)), -1)

        # Lift the dimension of coordinate embedding
        xy_local_u = self.xy_lift1(xy)   # (B,M,2F)
        xy_local_v = self.xy_lift2(xy)   # (B,M,2F)

        # Combine with global embedding (multiplication)
        xy_global_u = self._combine_embeddings(xy_local_u, Domain_enc, coupling='mul')    # (B,M,2F)
        xy_global_v = self._combine_embeddings(xy_local_v, Domain_enc, coupling='mul')    # (B,M,2F)

        # Get the kernels (compute once, shared by both components)
        enc = self._encode_par(par, par_flag)

        # Predict u and v using unified _zip method
        u = self._zip(xy_global_u, par, par_flag, 0, enc)
        v = self._zip(xy_global_v, par, par_flag, 1, enc)

        return u, v


class PI_GANO_geo_plate(DCONBlendMixin, BaseNeuralOperator):
    """
    PI-GANO variant using high-level geometric features instead of point clouds.
    Uses parametric geometry representation (e.g., shape parameters) as input.
    """

    def __init__(self, config, geo_feature_dim):
        super().__init__(config)

        # Geometry encoder from high-level features
        self.geo_encoder = nn.Sequential(*build_mlp(geo_feature_dim, self.fc_dim, self.n_layer, self.fc_dim))

        # Branch network
        self.branch = nn.Sequential(*build_mlp(4, 2*self.fc_dim, self.n_layer, 2*self.fc_dim))

        # Coordinate lifting layers (separate for u and v)
        self.xy_lift1 = nn.Linear(2, self.fc_dim)
        self.xy_lift2 = nn.Linear(2, self.fc_dim)

        # Trunk networks for u and v components (4 layers each)
        self.FC = nn.ModuleList([
            nn.ModuleList(build_sequential_layers(2*self.fc_dim, 2*self.fc_dim, 4)),  # for u
            nn.ModuleList(build_sequential_layers(2*self.fc_dim, 2*self.fc_dim, 4)),  # for v
        ])

        # Activation
        self.act = nn.Tanh()

    def _encode_par(self, par, par_flag):
        """Compute the parameter encoding from branch network."""
        enc = self.branch(par)
        enc_masked = enc * par_flag.unsqueeze(-1)
        enc = torch.amax(enc_masked, 1, keepdim=True)
        return enc

    def forward(self, x_coor, y_coor, par, par_flag, shape_coor, shape_flag, geo_feature):
        '''
        model input:
            x_coor (B, M): x-axis coordinates of the collocation points
            y_coor (B, M): y-axis coordinates of the collocation points
            par (B, M', 4): boundary coordinates and function values (x, y, u, v)
            par_flag (B, M'): valid flags for boundary points
            shape_coor: not used (kept for API compatibility)
            shape_flag: not used (kept for API compatibility)
            geo_feature (B, geo_feature_dim): high-level geometric features

        model output:
            u (B, M): x-component of the vector field
            v (B, M): y-component of the vector field
        '''
        # Extract number of points
        B, mD = x_coor.shape

        # Forward to get the domain embedding from high-level features
        Domain_enc = self.geo_encoder(geo_feature).unsqueeze(1)    # (B,1,F)

        # Concat coordinates
        xy = torch.cat((x_coor.unsqueeze(-1), y_coor.unsqueeze(-1)), -1)

        # Lift the dimension of coordinate embedding
        xy_local_u = self.xy_lift1(xy)   # (B,M,F)
        xy_local_v = self.xy_lift2(xy)   # (B,M,F)

        # Combine with global embedding (concatenation)
        xy_global_u = torch.cat((xy_local_u, Domain_enc.repeat(1,mD,1)), -1)    # (B,M,2F)
        xy_global_v = torch.cat((xy_local_v, Domain_enc.repeat(1,mD,1)), -1)    # (B,M,2F)

        # Get the kernels (compute once, shared by both components)
        enc = self._encode_par(par, par_flag)

        # Predict u and v using mixin methods
        u = self._predict_head(xy_global_u, enc, self.FC[0])
        v = self._predict_head(xy_global_v, enc, self.FC[1])

        return u, v


class baseline_geo_plate(DCONBlendMixin, BaseNeuralOperator):
    """
    Baseline variant using high-level geometric features.
    Simplified version without final gated modulation (no mean reduction with gating).
    """

    def __init__(self, config, geo_feature_dim):
        super().__init__(config)

        # Geometry encoder from high-level features
        self.geo_encoder = nn.Sequential(*build_mlp(geo_feature_dim, self.fc_dim, self.n_layer, self.fc_dim))

        # Branch network
        self.branch = nn.Sequential(*build_mlp(4, 2*self.fc_dim, self.n_layer, 2*self.fc_dim))

        # Coordinate lifting layers (separate for u and v)
        self.xy_lift1 = nn.Linear(2, self.fc_dim)
        self.xy_lift2 = nn.Linear(2, self.fc_dim)

        # Trunk networks for u and v components (4 layers each)
        self.FC = nn.ModuleList([
            nn.ModuleList(build_sequential_layers(2*self.fc_dim, 2*self.fc_dim, 4)),  # for u
            nn.ModuleList(build_sequential_layers(2*self.fc_dim, 2*self.fc_dim, 4)),  # for v
        ])

        # Activation
        self.act = nn.Tanh()

    def _encode_par(self, par, par_flag):
        """Compute the parameter encoding from branch network."""
        enc = self.branch(par)
        enc_masked = enc * par_flag.unsqueeze(-1)
        enc = torch.amax(enc_masked, 1, keepdim=True)
        return enc

    def forward(self, x_coor, y_coor, par, par_flag, shape_coor, shape_flag, geo_feature):
        '''
        model input:
            x_coor (B, M): x-axis coordinates of the collocation points
            y_coor (B, M): y-axis coordinates of the collocation points
            par (B, M', 4): boundary coordinates and function values (x, y, u, v)
            par_flag (B, M'): valid flags for boundary points
            shape_coor: not used (kept for API compatibility)
            shape_flag: not used (kept for API compatibility)
            geo_feature (B, geo_feature_dim): high-level geometric features

        model output:
            u (B, M, 2F): x-component features (no final reduction)
            v (B, M, 2F): y-component features (no final reduction)
        '''
        # Extract number of points
        B, mD = x_coor.shape

        # Forward to get the domain embedding from high-level features
        Domain_enc = self.geo_encoder(geo_feature).unsqueeze(1)    # (B,1,F)

        # Concat coordinates
        xy = torch.cat((x_coor.unsqueeze(-1), y_coor.unsqueeze(-1)), -1)

        # Lift the dimension of coordinate embedding
        xy_local_u = self.xy_lift1(xy)   # (B,M,F)
        xy_local_v = self.xy_lift2(xy)   # (B,M,F)

        # Combine with global embedding (concatenation)
        xy_global_u = torch.cat((xy_local_u, Domain_enc.repeat(1,mD,1)), -1)    # (B,M,2F)
        xy_global_v = torch.cat((xy_local_v, Domain_enc.repeat(1,mD,1)), -1)    # (B,M,2F)

        # Get the kernels (compute once, shared by both components)
        enc = self._encode_par(par, par_flag)

        # Predict u and v without final gating / mean reduction
        u = self._predict_head(xy_global_u, enc, self.FC[0], final_gate=False, reduce_mean=False)
        v = self._predict_head(xy_global_v, enc, self.FC[1], final_gate=False, reduce_mean=False)

        return u, v


''' ------------------------- New model template -------------------------- '''

class New_model_plate(nn.Module):

    def __init__(self, config):
        super().__init__()


    def forward(self, x_coor, y_coor, par, par_flag, shape_coor, shape_flag):
        '''
        par: (B, M', 4)
        par_flag: (B, M')
        x_coor: (B, M)
        y_coor: (B, M)

        return u, v: (B, M), (B, M)
        '''
        u = None
        v = None

        return u, v
