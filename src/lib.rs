use core::f64;
use image::{ImageBuffer, Luma};
use std::{
    error::Error,
    path::{Path, PathBuf},
    sync::Arc,
};
use zarrs::{
    array::Array, filesystem::FilesystemStore, group::Group, node::NodePath, plugin::ZarrVersion,
};

pub fn open_zarr(root_path: PathBuf) -> Result<(), Box<dyn Error>> {
    let store = Arc::new(FilesystemStore::new(&root_path)?);
    let root_path = NodePath::root();
    let root_group = Group::open(store.clone(), root_path.as_str())?;
    let attributes = root_group.attributes();
    let crs_path = attributes
        .get("coordinates")
        .map(|v| v.as_str())
        .flatten()
        .map(|v| NodePath::new(&format!("/{}", v)))
        .transpose()?;

    // Get WKT for projection information
    if let Some(path) = crs_path.clone() {
        let array = Array::open(store.clone(), path.as_str())?;
        let attributes = array.attributes();
        let _crs_wkt = attributes.get("crs_wkt");
    }

    // Filter all non-dimension and non-crs variables
    let children = root_group.child_arrays()?;
    for array in &children {
        let is_data_var = array_is_data_variable(array)?;
        if is_data_var && !array_is_crs(array, &crs_path) {
            let spatial_coords = get_spatial_coords(array);
            for &spatial_coord_name in &spatial_coords {
                let array = Array::open(store.clone(), &format!("/{}", spatial_coord_name))?;
                let chunk: ndarray::ArrayD<i64> = array.retrieve_chunk(&[0])?;

                let idx = find_spatial_coord_index(&array, 150_000.0)?;
                println!("{idx:?}");
            }

            let non_spatial_dims = get_non_spatial_dims(array, &spatial_coords)?;
            // println!("{non_spatial_dims:?}");
        }
    }

    let array = Array::open(store, "/x")?;
    let chunk: ndarray::ArrayD<i64> = array.retrieve_chunk(&[0])?;
    // println!("{chunk}");

    Ok(())
}

/// Check if the array contains actual data instead of coordinate values
fn array_is_data_variable(array: &Array<FilesystemStore>) -> Result<bool, Box<dyn Error>> {
    let array_path = array.path().as_path();
    let dim_names = array
        .dimension_names()
        .as_ref()
        .ok_or("Could not retrieve dimension names")?;

    let is_coord = dim_names.iter().any(|dim_name| {
        if let Some(dim) = dim_name
            && array_path.ends_with(dim)
        {
            true
        } else {
            false
        }
    });

    Ok(!is_coord)
}

/// Check if array contains coordinate system information
fn array_is_crs(array: &Array<FilesystemStore>, crs_path: &Option<NodePath>) -> bool {
    let crs_path = crs_path
        .as_ref()
        .map(|p| p.as_path())
        .unwrap_or(Path::new(""));
    let array_path = array.path().as_path();
    crs_path == array_path
}

/// Get all non-spatial dimensions of the Zarr array
fn get_non_spatial_dims(
    array: &Array<FilesystemStore>,
    spatial_dims: &Vec<&str>,
) -> Result<Vec<String>, Box<dyn Error>> {
    let mut non_spatial = vec![];
    let dim_names = array
        .dimension_names()
        .as_ref()
        .ok_or("Could not retrieve dimension names")?;
    for dim_name in dim_names {
        if let Some(name) = dim_name {
            let spatial = spatial_dims.iter().any(|&spatial_dim| spatial_dim == name);
            if !spatial {
                non_spatial.push(name.clone());
            }
        }
    }

    Ok(non_spatial)
}

fn get_spatial_coords(array: &Array<FilesystemStore>) -> Vec<&str> {
    array
        .attributes()
        .get("coordinates")
        .map(|v| v.as_str())
        .flatten()
        .map(|s| s.split(" ").collect::<Vec<_>>())
        .unwrap_or(vec![])
}

fn find_spatial_coord_index(
    array: &Array<FilesystemStore>,
    coord: f64,
) -> Result<usize, Box<dyn Error>> {
    let data_type = array.data_type();
    let name = data_type.name(ZarrVersion::V3);

    // Match the datatype of the input data
    let data = match name.as_deref() {
        Some("int64") => {
            array.retrieve_array_subset::<ndarray::Array1<i64>>(&array.subset_all())?
        }
        _ => unimplemented!("Data type not implemented yet"),
    };

    // Binary search to find the index
    let idx = data.as_slice().unwrap().binary_search_by(|&probe| {
        // compare the i64 probe (converted to f64) to our target
        (probe as f64).partial_cmp(&coord).unwrap()
    });

    let closest_idx = match idx {
        Ok(exact) => exact, // Found the exact coordinate
        Err(insertion_idx) => {
            // insertion_idx is where the target would be placed
            // check the neighbors to see which one is closer
            if insertion_idx == 0 {
                0
            } else if insertion_idx >= data.len() {
                data.len() - 1
            } else {
                let left = data[insertion_idx - 1] as f64;
                let right = data[insertion_idx] as f64;
                if (coord - left).abs() < (right - coord).abs() {
                    insertion_idx - 1
                } else {
                    insertion_idx
                }
            }
        }
    };

    Ok(closest_idx)
}

const TILE_PIXELS: u32 = 256;
const SOURCE_CRS: &str = "EPSG:3857";

#[derive(Debug)]
pub struct Tile {
    zoom: u8,
    x: u32,
    y: u32,
}

impl Tile {
    fn new(zoom: u8, x: u32, y: u32) -> Self {
        Tile { zoom, x, y }
    }
}

/// Transform tile from source to target system
fn sample_data_var(
    tile: &Tile,
    store: Arc<FilesystemStore>,
    path_x: &str,
    path_y: &str,
    data_var: &Array<FilesystemStore>,
) -> Result<(), Box<dyn Error>> {
    let array_x = Array::open(store.clone(), path_x)?;
    let data_type = array_x.data_type();
    let name = data_type.name(ZarrVersion::V3);
    let data_x = match name.as_deref() {
        Some("int64") => array_x.retrieve_array_subset::<ndarray::Array1<i64>>(&[0..2])?,
        _ => unimplemented!("Data type not implemented yet"),
    };

    let array_y = Array::open(store, path_y)?;
    let data_type = array_y.data_type();
    let name = data_type.name(ZarrVersion::V3);
    let data_y = match name.as_deref() {
        Some("int64") => array_y.retrieve_array_subset::<ndarray::Array1<i64>>(&[0..2])?,
        _ => unimplemented!("Data type not implemented yet"),
    };

    // Calculate origin and spacing for source
    let x_0 = data_x[0] as f64;
    let spacing_x = (data_x[1] - data_x[0]) as f64;

    let y_0 = data_y[0] as f64;
    let spacing_y = (data_y[1] - data_y[0]) as f64;

    let to = "EPSG:31287";
    let transformer = proj::Proj::try_from((SOURCE_CRS, to))?;

    let tile = tile_bbox(tile.x, tile.y, tile.zoom);
    let x_min_source = tile[0];
    let y_min_source = tile[1];
    let x_max_source = tile[2];
    let y_max_source = tile[3];

    // Calculate grid spacings
    let spacing_x_source = (x_max_source - x_min_source) / f64::from(TILE_PIXELS);
    let spacing_y_source = (y_max_source - y_min_source) / f64::from(TILE_PIXELS);

    // Transform from Web Mercator to target system
    let (x_min_target, y_min_target) = transformer.convert((x_min_source, y_min_source))?;
    let (x_max_target, y_max_target) = transformer.convert((x_max_source, y_max_source))?;

    // Index calculations for bounding box
    let min_ix = ((x_min_target - x_0) / spacing_x).floor() as u64;
    let mut min_iy = ((y_min_target - y_0) / spacing_y).floor() as u64;
    let max_ix = ((x_max_target - x_0) / spacing_x).ceil() as u64;
    let mut max_iy = ((y_max_target - y_0) / spacing_y).ceil() as u64;
    println!("max ix: {max_ix:?}");

    let shape_x = array_x.shape()[0];
    let min_ix = min_ix.saturating_sub(1);
    let max_ix = (max_ix + 1).min(shape_x - 1);

    if min_ix > max_ix || max_ix >= shape_x {
        return Err("Index for x coordinate out of bounds".into());
    }

    let shape_y = array_y.shape()[0];
    if min_iy > max_iy {
        let temp = min_iy;
        min_iy = max_iy;
        max_iy = temp;
    }
    if min_iy > max_iy || max_iy >= shape_y {
        return Err("Index for y coordinate out of bounds".into());
    }

    // Calculate the bounding box indices - ToDo: time!
    let x_range = min_ix..max_ix + 1;
    let y_range = min_iy..max_iy + 1;

    let tile_data =
        data_var.retrieve_array_subset::<ndarray::Array3<f32>>(&[0..1, y_range, x_range])?;

    // 256x256 tiles by default -> ToDo: make configurable
    let width = max_ix - min_ix;
    let height = max_iy - min_iy;
    let mut min = f32::INFINITY;
    let mut max = f32::NEG_INFINITY;
    let mut sampled_data = [[0.0; TILE_PIXELS as usize]; TILE_PIXELS as usize];
    for i in 0..TILE_PIXELS {
        for j in 0..TILE_PIXELS {
            let x = x_min_source + (f64::from(i) + 0.5) * spacing_x_source;
            let y = y_min_source + (f64::from(j) + 0.5) * spacing_y_source;

            // Transform from Web Mercator to target system
            let (xt, yt) = transformer.convert((x, y))?;

            // Index calculation
            if i == 255 && j == 0 {
                let test = ((xt - x_0) / spacing_x).round();
                println!("x_max: {x_max_source:?}");
                println!("before trafo: {x:?}");

                println!("x_max_trafo: {x_max_target:?}");
                println!("before trafo: {xt:?}");
                println!("compare: {test}");
            }
            let ix = ((xt - x_0) / spacing_x).round() as u64 - min_ix;
            let iy = ((yt - y_0) / spacing_y).round() as u64 - min_iy;

            if j == 0 && i == 255 {
                println!("Range x: {width:?}");
                println!("Range y: {height:?}");
                println!("Sampled index x : {ix:?}");
                println!("Sampled index y : {iy:?}");
            }

            // Bounds check and sample
            if ix <= width && iy <= height {
                let value = tile_data[[0, iy as usize, ix as usize]];
                if value < min {
                    min = value;
                }
                if value > max {
                    max = value;
                }
                sampled_data[i as usize][j as usize] = value;
            }
        }
    }

    // Create a new ImageBuffer.
    let mut img = ImageBuffer::new(TILE_PIXELS, TILE_PIXELS);

    // Iterate over the array and convert f64 to u8
    for (y, row) in sampled_data.iter().enumerate() {
        for (x, &value) in row.iter().enumerate() {
            // Normalize
            let val_norm = (value - min) / (max - min);

            // Scale to 0-255 and clamp to prevent overflow.
            let pixel_val = (val_norm * 255.0).clamp(0.0, 255.0) as u8;

            if y == 255 && x == 0 {
                println!("{value:?}");
            }

            img.put_pixel(x as u32, (TILE_PIXELS - 1) - y as u32, Luma([pixel_val]));
        }
    }

    // Save the image
    img.save("output.png")?;

    Ok(())
}

const EARTH_CIRCUMFERENCE: f64 = 40_075_016.685_578_5;

pub fn tile_bbox(x: u32, y: u32, zoom: u8) -> [f64; 4] {
    let tile_length = EARTH_CIRCUMFERENCE / f64::from(1_u32 << zoom);
    let min_x = EARTH_CIRCUMFERENCE * -0.5 + x as f64 * tile_length;
    let max_y = EARTH_CIRCUMFERENCE * 0.5 - y as f64 * tile_length;

    [min_x, max_y - tile_length, min_x + tile_length, max_y]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_open_zarr() {
        let path = PathBuf::from(r".\tests\fixtures\latest");
        let _ = open_zarr(path);
    }

    // 10/558/356
    // 9/275/177
    #[test]
    fn test_sample_tile() {
        let path = PathBuf::from(r".\tests\fixtures\latest");
        let store = Arc::new(FilesystemStore::new(&path).unwrap());
        let tile = Tile::new(9, 275, 177);
        let data_var = Array::open(store.clone(), "/airtemp").unwrap();
        let res = sample_data_var(&tile, store, "/x", "/y", &data_var);
        println!("{res:?}");
        assert!(res.is_ok());
    }
}
