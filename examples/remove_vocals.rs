//! Example: Remove vocals from an audio file
//! 
//! Usage: cargo run --example remove_vocals -- input.mp3 [output_dir]

use stem_splitter_core::SplitProgress;

fn main() -> anyhow::Result<()> {
    let mut args = std::env::args().skip(1);
    let input = args.next().unwrap_or_else(|| "2.mp3".into());
    let out = args.next().unwrap_or_else(|| "./out".into());

    stem_splitter_core::set_download_progress_callback(|d, t| {
        let pct = if t > 0 {
            (d as f64 / t as f64 * 100.0).round() as u64
        } else {
            0
        };
        if t > 0 {
            eprint!("\rModel: {:>3}% ({}/{})", pct, d, t);
            if d >= t {
                eprintln!();
            }
        } else {
            eprint!("\rModel: {} bytes", d);
        }
    });

    stem_splitter_core::set_split_progress_callback(|p| match p {
        SplitProgress::Stage(s) => {
            eprintln!("> {}", s);
        }
        SplitProgress::Chunks { done, total, percent } => {
            eprint!("\rSplit: {}/{} ({:.0}%)", done, total, percent);
            if done >= total {
                eprintln!();
            }
        }
        SplitProgress::Writing { ref stem, done, total, percent } => {
            eprintln!("Writing {}: {}/{} ({:.0}%)", stem, done, total, percent);
        }
        SplitProgress::Finished => {
            eprintln!("Finished.");
        }
    });

    let opts = stem_splitter_core::SplitOptions {
        output_dir: out,
        model_name: "htdemucs_ort_v1".into(),
        manifest_url_override: None,
        model_path: None,
    };

    // Use remove_vocals instead of split_file
    let res = stem_splitter_core::remove_vocals(&input, opts)?;
    
    eprintln!("\nDone:");
    eprintln!("  Instrumental (no vocals): {}", res.instrumental_path);
    eprintln!("  Vocals only: {}", res.vocals_path);
    
    Ok(())
}
