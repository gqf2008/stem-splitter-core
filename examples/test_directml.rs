//! Test DirectML initialization with ONNX Runtime
//! 
//! Run with: cargo run --no-default-features --features directml --example test_directml

use ort::{
    execution_providers::{DirectMLExecutionProvider, ExecutionProvider},
    session::builder::{GraphOptimizationLevel, SessionBuilder},
};

fn main() {
    // Initialize ORT
    println!("Initializing ONNX Runtime...");
    ort::init().commit().expect("Failed to init ORT");
    
    // Check DirectML availability
    println!("\nChecking DirectML providers...");
    for device_id in 0..8 {
        let provider = DirectMLExecutionProvider::default().with_device_id(device_id);
        match provider.is_available() {
            Ok(true) => println!("  Device {}: AVAILABLE", device_id),
            Ok(false) => println!("  Device {}: not available", device_id),
            Err(e) => println!("  Device {}: error - {:?}", device_id, e),
        }
    }
    
    // Try to get the model path
    let cache_dir = directories::BaseDirs::new()
        .map(|d| d.data_local_dir().join("StemSplitter/stem-splitter-core/cache/models"))
        .expect("Could not get cache dir");
    
    let model_files: Vec<_> = std::fs::read_dir(&cache_dir)
        .into_iter()
        .flatten()
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().map_or(false, |ext| ext == "ort"))
        .collect();
    
    if model_files.is_empty() {
        println!("\nNo .ort model found in {:?}", cache_dir);
        println!("Run `cargo run --example ensure_model` first.");
        return;
    }
    
    let model_path = model_files[0].path();
    println!("\nFound model: {:?}", model_path);
    
    // Try loading with DirectML
    println!("\nAttempting to load model with DirectML...");
    
    let dml_provider = DirectMLExecutionProvider::default()
        .with_device_id(0);
    
    let result = SessionBuilder::new()
        .and_then(|b| b.with_optimization_level(GraphOptimizationLevel::Level3))
        .and_then(|b| b.with_execution_providers([dml_provider.build()]))
        .and_then(|b| b.commit_from_file(&model_path));
    
    match result {
        Ok(session) => {
            println!("SUCCESS! Model loaded with DirectML.");
            println!("  Inputs: {:?}", session.inputs.iter().map(|i| &i.name).collect::<Vec<_>>());
            println!("  Outputs: {:?}", session.outputs.iter().map(|o| &o.name).collect::<Vec<_>>());
        }
        Err(e) => {
            println!("FAILED to load model with DirectML:");
            println!("  Error: {}", e);
            println!("  Debug: {:?}", e);
            
            // Try CPU fallback
            println!("\nTrying CPU fallback...");
            let cpu_result = SessionBuilder::new()
                .and_then(|b| b.with_optimization_level(GraphOptimizationLevel::Level3))
                .and_then(|b| b.with_intra_threads(4))
                .and_then(|b| b.commit_from_file(&model_path));
            
            match cpu_result {
                Ok(session) => {
                    println!("CPU session loaded successfully.");
                    println!("  Inputs: {:?}", session.inputs.iter().map(|i| &i.name).collect::<Vec<_>>());
                }
                Err(e) => {
                    println!("CPU also failed: {:?}", e);
                }
            }
        }
    }
}
