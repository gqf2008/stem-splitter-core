//! Example: Advanced separation with full control
//! 
//! Usage: cargo run --example custom_mix -- input.mp3 [output_dir]

use stem_splitter_core::{Separator, Stem, SplitOptions, SplitProgress};

fn main() -> anyhow::Result<()> {
    let mut args = std::env::args().skip(1);
    let input = args.next().unwrap_or_else(|| "1.mp3".into());
    let out_dir = args.next().unwrap_or_else(|| "./out".into());

    // Setup progress callbacks
    stem_splitter_core::set_download_progress_callback(|d, t| {
        if t > 0 {
            let pct = (d as f64 / t as f64 * 100.0).round() as u64;
            eprint!("\rModel: {:>3}%", pct);
            if d >= t { eprintln!(); }
        }
    });

    stem_splitter_core::set_split_progress_callback(|p| {
        if let SplitProgress::Stage(s) = p {
            eprintln!("> {}", s);
        }
    });

    let opts = SplitOptions {
        output_dir: out_dir.clone(),
        ..Default::default()
    };

    // Use the advanced Separator API
    eprintln!("Separating audio...");
    let stems = Separator::separate(&input, opts)?;

    eprintln!("\nSeparation complete! Sample rate: {}Hz, Samples: {}", 
              stems.sample_rate, stems.num_samples);

    // Create output directory
    std::fs::create_dir_all(&out_dir)?;

    // Example 1: Save individual stems
    eprintln!("\n--- Saving individual stems ---");
    stems.save(Stem::Vocals, &format!("{}/vocals.wav", out_dir))?;
    eprintln!("Saved: vocals.wav");
    
    stems.save(Stem::Drums, &format!("{}/drums.wav", out_dir))?;
    eprintln!("Saved: drums.wav");

    // Example 2: Save instrumental (everything except vocals)
    eprintln!("\n--- Saving instrumental (no vocals) ---");
    stems.save_mix_except(&[Stem::Vocals], &format!("{}/instrumental.wav", out_dir))?;
    eprintln!("Saved: instrumental.wav");

    // Example 3: Save custom mixes
    eprintln!("\n--- Saving custom mixes ---");
    
    // Rhythm section: drums + bass
    stems.save_mix(&[Stem::Drums, Stem::Bass], &format!("{}/rhythm.wav", out_dir))?;
    eprintln!("Saved: rhythm.wav (drums + bass)");
    
    // Melody: vocals + other
    stems.save_mix(&[Stem::Vocals, Stem::Other], &format!("{}/melody.wav", out_dir))?;
    eprintln!("Saved: melody.wav (vocals + other)");
    
    // Karaoke background: drums + bass + other (same as instrumental)
    stems.save_mix(&[Stem::Drums, Stem::Bass, Stem::Other], &format!("{}/karaoke.wav", out_dir))?;
    eprintln!("Saved: karaoke.wav (drums + bass + other)");

    // Example 4: Get raw audio data for custom processing
    eprintln!("\n--- Raw audio data access ---");
    let vocals_data = stems.get(Stem::Vocals);
    eprintln!("Vocals: {} samples (interleaved stereo)", vocals_data.len());
    
    let instrumental_audio = stems.mix_except_audio(&[Stem::Vocals]);
    eprintln!("Instrumental AudioData: {} samples, {}Hz, {} channels", 
              instrumental_audio.samples.len(),
              instrumental_audio.sample_rate,
              instrumental_audio.channels);

    eprintln!("\nDone! All files saved to: {}", out_dir);

    Ok(())
}
