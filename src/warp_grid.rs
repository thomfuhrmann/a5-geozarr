use std::collections::HashMap;

use a5::{LonLat, cell_to_lonlat};

use crate::{error::ZarrError, geozarr::SpatialRegistration};

pub const TARGET_CRS: &'static str = "EPSG:4326";

#[derive(Debug, Clone)]
pub struct WarpGrid {
    pub grid: HashMap<u64, (i64, i64)>,
    pub bbox: [u64; 4],
}

impl WarpGrid {
    pub fn new(grid: HashMap<u64, (i64, i64)>, bbox: [u64; 4]) -> Self {
        Self { grid, bbox }
    }

    pub(crate) fn calculate_warp_grid(
        cell_ids: &[u64],
        crs: &str,
        shape: [u64; 2],
        spatial_transform: [f64; 6],
        spatial_registration: &SpatialRegistration,
    ) -> Result<Option<Self>, ZarrError> {
        // inverse affine transformation from spatial coordinate system to pixel grid
        let src_inv_affine = inverse_affine(spatial_transform);

        // inverse spatial coordinate transformation
        let inv_trafo =
            proj::Proj::try_from((TARGET_CRS, crs)).map_err(ZarrError::ProjCreateError)?;

        let mut indices = HashMap::with_capacity(cell_ids.len());

        let mut tile_grid_left = i64::MAX;
        let mut tile_grid_right = i64::MIN;
        let mut tile_grid_bottom = i64::MIN;
        let mut tile_grid_top = i64::MAX;

        #[allow(clippy::cast_precision_loss)]
        let width = shape[1] as f64;
        #[allow(clippy::cast_precision_loss)]
        let height = shape[0] as f64;

        let src_uses_0_360 = spatial_transform[2] > 180.0
            || (spatial_transform[2] + spatial_transform[0] * shape[1] as f64) > 180.0;

        let mut has_valid_pixel = false;

        for &cell_id in cell_ids {
            let LonLat {
                longitude,
                latitude,
            } = cell_to_lonlat(cell_id).map_err(|e| ZarrError::ConvertLonLatError(e))?;

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
            };

            let down = match spatial_registration {
                SpatialRegistration::Pixel => {
                    (src_inv_affine[3] * x_src + src_inv_affine[4] * y_src + src_inv_affine[5])
                        .floor()
                }
                SpatialRegistration::Node => {
                    (src_inv_affine[3] * x_src + src_inv_affine[4] * y_src + src_inv_affine[5])
                        .round()
                }
            };

            #[allow(clippy::cast_possible_truncation)]
            let right = if (0.0..width).contains(&right) {
                right as i64
            } else {
                -1
            };

            #[allow(clippy::cast_possible_truncation)]
            let down = if (0.0..height).contains(&down) {
                down as i64
            } else {
                -1
            };

            if right >= 0 && down >= 0 {
                has_valid_pixel = true;
                tile_grid_left = tile_grid_left.min(right);
                tile_grid_right = tile_grid_right.max(right);
                tile_grid_top = tile_grid_top.min(down);
                tile_grid_bottom = tile_grid_bottom.max(down);
            }

            indices.insert(cell_id, (right, down));
        }

        // tile grid lies outside source data
        if !has_valid_pixel {
            return Ok(None);
        }

        Ok(Some(WarpGrid::new(
            indices,
            [
                tile_grid_left.cast_unsigned(),
                tile_grid_top.cast_unsigned(),
                tile_grid_right.cast_unsigned(),
                tile_grid_bottom.cast_unsigned(),
            ],
        )))
    }
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

#[cfg(test)]
mod tests {
    use crate::warp_grid::inverse_affine;

    #[test]
    fn test_inverse_affine() {
        let transform = [2.0, 0.0, 10.0, 0.0, 3.0, 20.0];
        let inverse = inverse_affine(transform);

        let x = inverse[0] * 12.0 + inverse[1] * 23.0 + inverse[2];
        let y = inverse[3] * 12.0 + inverse[4] * 23.0 + inverse[5];

        assert!((x - 1.0).abs() < 1e-12);
        assert!((y - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_warp_grid_size() {
        let (warp_grid, _warp_grid_bbox) = calculate_warp_grid_for_bbox(
            BBox::new(0.0, 0.0, 4.0, 4.0),
            "EPSG:4326",
            "EPSG:4326",
            [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            [4, 4],
            4,
            &SpatialRegistration::Pixel,
        )
        .expect("could not calculate warp grid")
        .expect("should be some");

        assert_eq!(warp_grid.len(), 32);
    }

    #[test]
    fn test_warp_grid_identity() {
        let result = calculate_warp_grid_for_bbox(
            BBox::new(0.0, 0.0, 4.0, 4.0),
            "EPSG:4326",
            "EPSG:4326",
            [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            [4, 4],
            4,
            &SpatialRegistration::Pixel,
        )
        .expect("could not calculate warp grid");

        assert_eq!(
            result,
            Some((
                vec![
                    0, 0, 0, 1, 0, 2, 0, 3, 1, 0, 1, 1, 1, 2, 1, 3, 2, 0, 2, 1, 2, 2, 2, 3, 3, 0,
                    3, 1, 3, 2, 3, 3
                ]
                .into_boxed_slice(),
                [0, 3, 3, 0]
            ))
        );
    }

    #[test]
    fn test_warp_grid_outside() {
        let result = calculate_warp_grid_for_bbox(
            BBox::new(5.0, 10.0, 7.0, 12.0),
            "EPSG:4326",
            "EPSG:4326",
            [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            [4, 4],
            4,
            &SpatialRegistration::Pixel,
        )
        .expect("could not calculate warp grid");

        assert!(result.is_none());
    }

    #[test]
    fn test_warp_grid_partial() {
        let result = calculate_warp_grid_for_bbox(
            BBox::new(2.0, 2.0, 5.0, 5.0),
            "EPSG:4326",
            "EPSG:4326",
            [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            [4, 4],
            4,
            &SpatialRegistration::Pixel,
        )
        .expect("could not calculate warp grid");

        assert_eq!(
            result,
            Some((
                vec![
                    2, 2, 2, 3, 2, 3, 2, -1, 3, 2, 3, 3, 3, 3, 3, -1, 3, 2, 3, 3, 3, 3, 3, -1, -1,
                    2, -1, 3, -1, 3, -1, -1
                ]
                .into_boxed_slice(),
                [2, 3, 3, 2]
            ))
        );
    }
}
