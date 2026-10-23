use std::collections::BTreeMap;

use a5::{LonLat, cell_to_lonlat};

use crate::{
    error::ZarrError,
    geozarr::{GeozarrMetadata, SpatialRegistration},
};

pub const TARGET_CRS: &str = "EPSG:4326";

#[derive(Debug, Clone)]
pub struct WarpGrid {
    pub grid: BTreeMap<u64, (i64, i64)>,
    pub grid_bbox: Option<[u64; 4]>,
}

impl WarpGrid {
    pub fn new(grid: BTreeMap<u64, (i64, i64)>, grid_bbox: Option<[u64; 4]>) -> Self {
        Self { grid, grid_bbox }
    }

    pub fn calculate_warp_grid(
        cell_ids: &[u64],
        geozarr_metadata: &GeozarrMetadata,
    ) -> Result<Self, ZarrError> {
        let spatial_transform =
            geozarr_metadata
                .spatial_transform
                .ok_or(ZarrError::GeoZarrMetadataError(
                    "Missing spatial:transform".into(),
                ))?;

        let spatial_shape =
            geozarr_metadata
                .spatial_shape
                .ok_or(ZarrError::GeoZarrMetadataError(
                    "Missing spatial:shape".into(),
                ))?;

        let spatial_registration = &geozarr_metadata.spatial_registration;

        let crs_str = &geozarr_metadata
            .proj
            .clone()
            .map(|proj| proj.into_string())
            .ok_or(ZarrError::GeoZarrMetadataError("Missing proj".into()))?[..];

        // inverse affine transformation from spatial coordinate system to pixel grid
        let src_inv_affine = Self::inverse_affine(spatial_transform);

        // inverse spatial coordinate transformation
        let inv_trafo =
            proj::Proj::try_from((TARGET_CRS, crs_str)).map_err(ZarrError::ProjCreateError)?;

        let mut indices = BTreeMap::new();

        let mut cell_grid_left = i64::MAX;
        let mut cell_grid_right = i64::MIN;
        let mut cell_grid_bottom = i64::MIN;
        let mut cell_grid_top = i64::MAX;

        // TODO: check for geographical crs based on metadata
        let src_uses_0_360 = crs_str == "EPSG:4326"
            && (spatial_transform[2] > 180.0
                || (spatial_transform[2] + spatial_transform[0] * spatial_shape[1] as f64) > 180.0);

        let mut has_valid_pixel = false;

        for &cell_id in cell_ids {
            let LonLat {
                longitude,
                latitude,
            } = cell_to_lonlat(cell_id).map_err(ZarrError::ConvertLonLatError)?;

            let x_target = longitude.0;
            let y_target = latitude.0;

            let (mut x_src, y_src) = inv_trafo
                .convert((x_target, y_target))
                .map_err(ZarrError::ProjError)?;

            if src_uses_0_360 && x_src < 0.0 {
                x_src += 360.0;
            }

            let right = match spatial_registration {
                SpatialRegistration::Pixel => {
                    (src_inv_affine[0] * x_src + src_inv_affine[1] * y_src + src_inv_affine[2])
                        .floor()
                }
                SpatialRegistration::Node => {
                    (src_inv_affine[0] * x_src + src_inv_affine[1] * y_src + src_inv_affine[2])
                        .round()
                }
            } as i64;

            let down = match spatial_registration {
                SpatialRegistration::Pixel => {
                    (src_inv_affine[3] * x_src + src_inv_affine[4] * y_src + src_inv_affine[5])
                        .floor()
                }
                SpatialRegistration::Node => {
                    (src_inv_affine[3] * x_src + src_inv_affine[4] * y_src + src_inv_affine[5])
                        .round()
                }
            } as i64;

            let width = spatial_shape[1] as i64;
            let height = spatial_shape[0] as i64;
            if (0..width).contains(&right) && (0..height).contains(&down) {
                has_valid_pixel = true;
                cell_grid_left = cell_grid_left.min(right);
                cell_grid_right = cell_grid_right.max(right);
                cell_grid_top = cell_grid_top.min(down);
                cell_grid_bottom = cell_grid_bottom.max(down);
            }

            indices.insert(cell_id, (right, down));
        }

        // tile grid lies outside source data
        if !has_valid_pixel {
            return Ok(WarpGrid::new(indices, None));
        }

        Ok(WarpGrid::new(
            indices,
            Some([
                cell_grid_left.cast_unsigned(),
                cell_grid_top.cast_unsigned(),
                cell_grid_right.cast_unsigned(),
                cell_grid_bottom.cast_unsigned(),
            ]),
        ))
    }

    /// Calculates inverse affine transformation
    fn inverse_affine(transform: [f64; 6]) -> [f64; 6] {
        let src_a = transform[0];
        let src_b = transform[1];
        let src_c = transform[2];
        let src_d = transform[3];
        let src_e = transform[4];
        let src_f = transform[5];
        let det = src_a * src_e - src_b * src_d;
        [
            src_e / det,
            -src_b / det,
            (src_b * src_f - src_e * src_c) / det,
            -src_d / det,
            src_a / det,
            (src_d * src_c - src_a * src_f) / det,
        ]
    }
}

#[cfg(test)]
mod tests {
    use std::sync::LazyLock;

    use crate::{geozarr::Proj, test_utils::read_cell_ids};

    use super::*;

    pub(crate) static CELL_IDS: LazyLock<Vec<u64>> = LazyLock::new(|| {
        read_cell_ids("tests/fixtures/a5_15_hzb_fid_13987.csv").expect("should read csv")
    });

    #[test]
    fn test_inverse_affine() {
        let transform = [2.0, 0.0, 10.0, 0.0, 3.0, 20.0];
        let inverse = WarpGrid::inverse_affine(transform);

        let x = inverse[0] * 12.0 + inverse[1] * 23.0 + inverse[2];
        let y = inverse[3] * 12.0 + inverse[4] * 23.0 + inverse[5];

        assert!((x - 1.0).abs() < 1e-12);
        assert!((y - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_warp_grid_size() {
        let metadata = GeozarrMetadata::new()
            .with_proj(Some(Proj::Code("EPSG:4326".into())))
            .with_spatial_shape(Some([50, 20]))
            .with_spatial_transform(Some([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]));
        let warp_grid = WarpGrid::calculate_warp_grid(&CELL_IDS, &metadata)
            .expect("should calculate warp grid");
        assert_eq!(warp_grid.grid.len(), CELL_IDS.len());
    }

    #[test]
    fn test_warp_grid_identity() {
        let metadata = GeozarrMetadata::new()
            .with_proj(Some(Proj::Code("EPSG:4326".into())))
            .with_spatial_shape(Some([50, 20]))
            .with_spatial_transform(Some([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]));
        let warp_grid = WarpGrid::calculate_warp_grid(&CELL_IDS, &metadata)
            .expect("should calculate warp grid");
        let exp = CELL_IDS
            .iter()
            .flat_map(|&cell_id| {
                cell_to_lonlat(cell_id).map(|lon_lat| {
                    (
                        lon_lat.longitude.0.floor() as i64,
                        lon_lat.latitude.0.floor() as i64,
                    )
                })
            })
            .collect::<Vec<_>>();
        assert_eq!(warp_grid.grid.into_values().collect::<Vec<_>>(), exp);
    }

    #[test]
    fn test_warp_grid_outside() {
        let metadata = GeozarrMetadata::new()
            .with_proj(Some(Proj::Code("EPSG:4326".into())))
            .with_spatial_shape(Some([10, 10]))
            .with_spatial_transform(Some([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]));
        let warp_grid = WarpGrid::calculate_warp_grid(&CELL_IDS, &metadata)
            .expect("should calculate warp grid");
        assert!(warp_grid.grid_bbox.is_none());
    }
}
