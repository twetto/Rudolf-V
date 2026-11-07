use pyo3::prelude::*;
use serde::de::DeserializeOwned;

mod checkerboard;
mod frontend;
mod stereo;

#[pymodule]
fn _rudolf_v(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<frontend::FrontendConfig>()?;
    m.add_class::<frontend::PyFrontend>()?;
    m.add_class::<stereo::PyStereo>()?;
    m.add_class::<checkerboard::PyCheckerboardDetector>()?;
    Ok(())
}

/// The `key` section of the YAML file at `path`, or None when the file has
/// no such section. The rest of the file is not read.
pub(crate) fn read_section<T: DeserializeOwned>(path: &str, key: &str) -> PyResult<Option<T>> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("Failed to read {path}: {e}")))?;
    let file: serde_yaml::Value = serde_yaml::from_str(&text).map_err(|e| {
        pyo3::exceptions::PyValueError::new_err(format!("Failed to parse {path}: {e}"))
    })?;
    let Some(section) = file.get(key) else {
        return Ok(None);
    };
    serde_yaml::from_value(section.clone())
        .map(Some)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("{path}: {key}: {e}")))
}

pub(crate) fn config_error(e: rudolf_v::config::ConfigError) -> PyErr {
    pyo3::exceptions::PyValueError::new_err(e.to_string())
}
