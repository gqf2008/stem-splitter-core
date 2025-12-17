//! Test long audio processing with chunking
//! 
//! Usage: cargo run --release --example test_long_audio

use stem_splitter_core::SplitProgress;

fn main() -> anyhow::Result<()> {
    let input = std::env::args().nth(1).unwrap_or_else(|| "95MB.mp3".into());
    let out = std::env::args().nth(2).unwrap_or_else(|| "./out_long".into());
    
    // Use cached model path directly to avoid network issues
    let model_path = r"C:\Users\DELL\AppData\Local\StemSplitter\stem-splitter-core\cache\models\HTDemucs-ORT-09dc1655.ort";

    stem_splitter_core::set_split_progress_callback(|p| match p {
        SplitProgress::Stage(s) => {
            eprintln!("> {}", s);
        }
        SplitProgress::Chunks { done, total, percent } => {
            eprint!("\rProcessing: {}/{} ({:.1}%)", done, total, percent);
            if done >= total {
                eprintln!();
            }
        }
        SplitProgress::Writing { ref stem, done, total, percent } => {
            eprintln!("  {}: {}/{} ({:.1}%)", stem, done, total, percent);
        }
        SplitProgress::Finished => {
            eprintln!("Finished!");
        }
    });

    let opts = stem_splitter_core::SplitOptions {
        output_dir: out,
        model_name: "htdemucs_ort_v1".into(),
        manifest_url_override: None,
        model_path: Some(model_path.to_string()),
        chunk_seconds: Some(60), // 3 minutes per chunk for testing
    };

    eprintln!("Processing: {}", input);
    eprintln!("Using model: {}", model_path);
    eprintln!("Chunk size: {} seconds", opts.chunk_seconds.unwrap_or(300));
    eprintln!();

    let start = std::time::Instant::now();
    let res = stem_splitter_core::remove_vocals(&input, opts)?;
    let elapsed = start.elapsed();

    eprintln!("\nDone in {:.1} seconds!", elapsed.as_secs_f64());
    eprintln!("  Instrumental: {}", res.instrumental_path);
    eprintln!("  Vocals: {}", res.vocals_path);
    
    Ok(())
}
