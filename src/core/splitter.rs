use crate::{
    core::{
        audio::{read_audio, write_audio},
        dsp::to_planar_stereo,
        engine,
    },
    error::Result,
    io::progress::{emit_split_progress, SplitProgress},
    model::model_manager::ensure_model,
    types::{AudioData, SplitOptions, SplitResult},
};

use std::{
    collections::HashMap,
    fs,
    path::{Path, PathBuf},
};
use tempfile::tempdir;

/// Result for vocal removal operation
#[derive(Clone, Debug)]
pub struct VocalRemovalResult {
    /// Path to instrumental track (everything except vocals)
    pub instrumental_path: String,
    /// Path to isolated vocals track
    pub vocals_path: String,
}

/// Available stem types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Stem {
    Vocals,
    Drums,
    Bass,
    Other,
}

impl Stem {
    /// Get all available stems
    pub fn all() -> &'static [Stem] {
        &[Stem::Vocals, Stem::Drums, Stem::Bass, Stem::Other]
    }

    /// Get stem name as string
    pub fn name(&self) -> &'static str {
        match self {
            Stem::Vocals => "vocals",
            Stem::Drums => "drums",
            Stem::Bass => "bass",
            Stem::Other => "other",
        }
    }
}

/// Separated audio stems - the result of audio separation.
/// 
/// This struct holds all separated stems in memory, allowing you to:
/// - Access individual stems
/// - Mix multiple stems together
/// - Save stems to files
/// 
/// # Example
/// ```no_run
/// use stem_splitter_core::{Separator, Stem, SplitOptions};
/// 
/// let opts = SplitOptions::default();
/// let stems = Separator::separate("song.mp3", opts)?;
/// 
/// // Get individual stem
/// let vocals = stems.get(Stem::Vocals);
/// 
/// // Mix drums + bass (rhythm section)
/// let rhythm = stems.mix(&[Stem::Drums, Stem::Bass]);
/// 
/// // Mix everything except vocals (instrumental)
/// let instrumental = stems.mix_except(&[Stem::Vocals]);
/// 
/// // Save to file
/// stems.save(Stem::Vocals, "vocals.wav")?;
/// stems.save_mix(&[Stem::Drums, Stem::Bass], "rhythm.wav")?;
/// # Ok::<(), anyhow::Error>(())
/// ```
#[derive(Clone)]
pub struct SeparatedStems {
    /// Raw stem data: [stem_index][sample_index] = [left, right]
    stems: HashMap<Stem, Vec<[f32; 2]>>,
    /// Sample rate (always 44100 for htdemucs)
    pub sample_rate: u32,
    /// Number of samples per channel
    pub num_samples: usize,
}

impl SeparatedStems {
    /// Get a single stem's audio data as interleaved stereo samples
    pub fn get(&self, stem: Stem) -> Vec<f32> {
        self.stems
            .get(&stem)
            .map(|data| {
                let mut out = Vec::with_capacity(data.len() * 2);
                for s in data {
                    out.push(s[0]);
                    out.push(s[1]);
                }
                out
            })
            .unwrap_or_default()
    }

    /// Get a single stem as AudioData
    pub fn get_audio(&self, stem: Stem) -> AudioData {
        AudioData {
            samples: self.get(stem),
            sample_rate: self.sample_rate,
            channels: 2,
        }
    }

    /// Mix multiple stems together
    pub fn mix(&self, stems: &[Stem]) -> Vec<f32> {
        let mut out = vec![0.0f32; self.num_samples * 2];
        for stem in stems {
            if let Some(data) = self.stems.get(stem) {
                for (i, s) in data.iter().enumerate() {
                    out[i * 2] += s[0];
                    out[i * 2 + 1] += s[1];
                }
            }
        }
        out
    }

    /// Mix multiple stems as AudioData
    pub fn mix_audio(&self, stems: &[Stem]) -> AudioData {
        AudioData {
            samples: self.mix(stems),
            sample_rate: self.sample_rate,
            channels: 2,
        }
    }

    /// Mix all stems except the specified ones
    pub fn mix_except(&self, exclude: &[Stem]) -> Vec<f32> {
        let include: Vec<Stem> = Stem::all()
            .iter()
            .copied()
            .filter(|s| !exclude.contains(s))
            .collect();
        self.mix(&include)
    }

    /// Mix all stems except the specified ones as AudioData
    pub fn mix_except_audio(&self, exclude: &[Stem]) -> AudioData {
        AudioData {
            samples: self.mix_except(exclude),
            sample_rate: self.sample_rate,
            channels: 2,
        }
    }

    /// Save a single stem to a WAV file
    pub fn save(&self, stem: Stem, path: &str) -> Result<()> {
        let audio = self.get_audio(stem);
        write_audio(path, &audio)?;
        Ok(())
    }

    /// Save a mix of multiple stems to a WAV file
    pub fn save_mix(&self, stems: &[Stem], path: &str) -> Result<()> {
        let audio = self.mix_audio(stems);
        write_audio(path, &audio)?;
        Ok(())
    }

    /// Save a mix of all stems except the specified ones to a WAV file
    pub fn save_mix_except(&self, exclude: &[Stem], path: &str) -> Result<()> {
        let audio = self.mix_except_audio(exclude);
        write_audio(path, &audio)?;
        Ok(())
    }
}

/// High-level separator for complete control over audio separation.
/// 
/// # Example
/// ```no_run
/// use stem_splitter_core::{Separator, Stem, SplitOptions};
/// 
/// // Separate audio
/// let stems = Separator::separate("song.mp3", SplitOptions::default())?;
/// 
/// // Save individual stems
/// stems.save(Stem::Vocals, "vocals.wav")?;
/// 
/// // Save instrumental (everything except vocals)
/// stems.save_mix_except(&[Stem::Vocals], "instrumental.wav")?;
/// 
/// // Save custom mix (drums + bass only)
/// stems.save_mix(&[Stem::Drums, Stem::Bass], "rhythm.wav")?;
/// 
/// // Get raw audio data for further processing
/// let vocals_data = stems.get(Stem::Vocals);
/// let instrumental_audio = stems.mix_except_audio(&[Stem::Vocals]);
/// # Ok::<(), anyhow::Error>(())
/// ```
pub struct Separator;

impl Separator {
    /// Separate an audio file into individual stems.
    /// 
    /// Returns `SeparatedStems` which provides full control over
    /// accessing, mixing, and saving the separated audio.
    pub fn separate(input_path: &str, opts: SplitOptions) -> Result<SeparatedStems> {
        let stem_data = separate_stems_internal(input_path, &opts)?;
        
        let mut stems = HashMap::new();
        
        let get_idx = |key: &str, fallback: usize| -> usize {
            stem_data.name_idx
                .get(key)
                .copied()
                .unwrap_or(fallback.min(stem_data.stems_count.saturating_sub(1)))
        };
        
        stems.insert(Stem::Vocals, stem_data.acc[get_idx("vocals", 0)].clone());
        stems.insert(Stem::Drums, stem_data.acc[get_idx("drums", 1)].clone());
        stems.insert(Stem::Bass, stem_data.acc[get_idx("bass", 2)].clone());
        stems.insert(Stem::Other, stem_data.acc[get_idx("other", 3)].clone());
        
        emit_split_progress(SplitProgress::Finished);
        
        Ok(SeparatedStems {
            stems,
            sample_rate: stem_data.sample_rate,
            num_samples: stem_data.n,
        })
    }
}

/// Internal struct holding separated stem data
struct StemDataInternal {
    acc: Vec<Vec<[f32; 2]>>,
    stems_count: usize,
    name_idx: HashMap<String, usize>,
    sample_rate: u32,
    n: usize,
}

/// Core separation logic - shared between all public APIs
fn separate_stems_internal(input_path: &str, opts: &SplitOptions) -> Result<StemDataInternal> {
    emit_split_progress(SplitProgress::Stage("resolve_model"));
    let handle = ensure_model(&opts.model_name, opts.manifest_url_override.as_deref())?;

    emit_split_progress(SplitProgress::Stage("engine_preload"));
    engine::preload(&handle)?;

    let mf = engine::manifest();

    if mf.sample_rate != 44100 {
        return Err(anyhow::anyhow!("Currently expecting 44.1k model").into());
    }

    emit_split_progress(SplitProgress::Stage("read_audio"));
    let audio = read_audio(input_path)?;
    let stereo = to_planar_stereo(&audio.samples, audio.channels);
    let n = stereo.len();

    if n == 0 {
        return Err(anyhow::anyhow!("Empty audio").into());
    }

    let win = mf.window;
    let hop = mf.hop;

    if !(win > 0 && hop > 0 && hop <= win) {
        return Err(anyhow::anyhow!("Bad win/hop in manifest").into());
    }

    if std::env::var("DEBUG_STEMS").is_ok() {
        eprintln!("Window settings: win={}, hop={}, overlap={}", win, hop, win - hop);
    }

    let stems_names = mf.stems.clone();
    let mut stems_count = stems_names.len().max(1);

    let mut left_raw = vec![0f32; win];
    let mut right_raw = vec![0f32; win];

    let mut acc: Vec<Vec<[f32; 2]>> = Vec::new();
    let mut pos = 0usize;
    let mut first_chunk = true;

    emit_split_progress(SplitProgress::Stage("infer"));
    while pos < n {
        for i in 0..win {
            let idx = pos + i;
            if idx < n {
                left_raw[i] = stereo[idx][0];
                right_raw[i] = stereo[idx][1];
            } else {
                left_raw[i] = 0.0;
                right_raw[i] = 0.0;
            }
        }

        let out = engine::run_window_demucs(&left_raw, &right_raw)?;
        let (s_count, _, t_out) = (out.shape()[0], out.shape()[1], out.shape()[2]);

        if first_chunk {
            stems_count = s_count;
            acc = vec![vec![[0f32; 2]; n]; stems_count];
            first_chunk = false;
        }

        let copy_len = hop.min(t_out).min(n - pos);
        for st in 0..stems_count {
            for i in 0..copy_len {
                acc[st][pos + i][0] = out[(st, 0, i)];
                acc[st][pos + i][1] = out[(st, 1, i)];
            }
        }

        if pos + hop >= n {
            break;
        }
        pos += hop;
    }

    let names = if stems_names.is_empty() {
        vec!["vocals".into(), "drums".into(), "bass".into(), "other".into()]
    } else {
        stems_names
    };

    let mut name_idx: HashMap<String, usize> = HashMap::new();
    for (i, name) in names.iter().enumerate() {
        name_idx.insert(name.to_lowercase(), i);
    }

    if std::env::var("DEBUG_STEMS").is_ok() {
        for st in 0..stems_count {
            let max_val = acc[st].iter()
                .map(|s| s[0].abs().max(s[1].abs()))
                .fold(0.0f32, f32::max);
            eprintln!("Accumulator [stem {}]: max_value={:.6}, samples={}", st, max_val, acc[st].len());
        }
    }

    Ok(StemDataInternal {
        acc,
        stems_count,
        name_idx,
        sample_rate: mf.sample_rate,
        n,
    })
}

/// Split an audio file into 4 separate stems: vocals, drums, bass, other
pub fn split_file(input_path: &str, opts: SplitOptions) -> Result<SplitResult> {
    let stem_data = separate_stems_internal(input_path, &opts)?;
    let StemDataInternal { acc, stems_count, name_idx, sample_rate, n } = stem_data;

    let tmp = tempdir()?;
    let tmp_dir = tmp.path().to_path_buf();

    fs::create_dir_all(&opts.output_dir)?;

    emit_split_progress(SplitProgress::Stage("write_stems"));

    let stem_to_wav = |st: usize, base: &str| -> Result<String> {
        let mut inter = Vec::with_capacity(n * 2);
        for sample in &acc[st][..n] {
            inter.push(sample[0]);
            inter.push(sample[1]);
        }

        emit_split_progress(SplitProgress::Writing {
            stem: base.to_string(),
            done: n,
            total: n,
            percent: 100.0,
        });

        let data = AudioData {
            samples: inter,
            sample_rate,
            channels: 2,
        };

        let p = tmp_dir.join(format!("{base}.wav"));
        write_audio(p.to_str().unwrap(), &data)?;
        Ok(p.to_string_lossy().into())
    };

    let get_idx = |key: &str, fallback: usize| -> usize {
        name_idx
            .get(key)
            .copied()
            .unwrap_or(fallback.min(stems_count.saturating_sub(1)))
    };

    let v_path = stem_to_wav(get_idx("vocals", 0), "vocals")?;
    let d_path = stem_to_wav(get_idx("drums", 1), "drums")?;
    let b_path = stem_to_wav(get_idx("bass", 2), "bass")?;
    let o_path = stem_to_wav(get_idx("other", 3), "other")?;

    emit_split_progress(SplitProgress::Stage("finalize"));

    let file_stem = Path::new(input_path)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("output");
    let base = PathBuf::from(&opts.output_dir).join(file_stem);

    let vocals_out = copy_to(&v_path, &format!("{}_vocals.wav", base.to_string_lossy()))?;
    let drums_out = copy_to(&d_path, &format!("{}_drums.wav", base.to_string_lossy()))?;
    let bass_out = copy_to(&b_path, &format!("{}_bass.wav", base.to_string_lossy()))?;
    let other_out = copy_to(&o_path, &format!("{}_other.wav", base.to_string_lossy()))?;

    emit_split_progress(SplitProgress::Finished);

    Ok(SplitResult {
        vocals_path: vocals_out,
        drums_path: drums_out,
        bass_path: bass_out,
        other_path: other_out,
    })
}

/// Remove vocals from an audio file, producing instrumental and vocals tracks.
/// 
/// This is more efficient if you only need the instrumental (karaoke) version,
/// as it only writes 2 files instead of 4.
/// 
/// # Example
/// ```no_run
/// use stem_splitter_core::{remove_vocals, SplitOptions};
/// 
/// let opts = SplitOptions::default();
/// let result = remove_vocals("song.mp3", opts)?;
/// println!("Instrumental: {}", result.instrumental_path);
/// println!("Vocals: {}", result.vocals_path);
/// # Ok::<(), anyhow::Error>(())
/// ```
pub fn remove_vocals(input_path: &str, opts: SplitOptions) -> Result<VocalRemovalResult> {
    let stem_data = separate_stems_internal(input_path, &opts)?;
    let StemDataInternal { acc, stems_count, name_idx, sample_rate, n } = stem_data;

    let tmp = tempdir()?;
    let tmp_dir = tmp.path().to_path_buf();

    fs::create_dir_all(&opts.output_dir)?;

    emit_split_progress(SplitProgress::Stage("write_stems"));

    let get_idx = |key: &str, fallback: usize| -> usize {
        name_idx
            .get(key)
            .copied()
            .unwrap_or(fallback.min(stems_count.saturating_sub(1)))
    };

    let vocals_idx = get_idx("vocals", 0);

    // Write vocals
    let mut vocals_samples = Vec::with_capacity(n * 2);
    for sample in &acc[vocals_idx][..n] {
        vocals_samples.push(sample[0]);
        vocals_samples.push(sample[1]);
    }

    emit_split_progress(SplitProgress::Writing {
        stem: "vocals".to_string(),
        done: n,
        total: n,
        percent: 100.0,
    });

    let vocals_data = AudioData {
        samples: vocals_samples,
        sample_rate,
        channels: 2,
    };
    let vocals_tmp = tmp_dir.join("vocals.wav");
    write_audio(vocals_tmp.to_str().unwrap(), &vocals_data)?;

    // Create instrumental (everything except vocals)
    let mut instrumental = Vec::with_capacity(n * 2);
    for i in 0..n {
        let mut left = 0.0f32;
        let mut right = 0.0f32;
        for st in 0..stems_count {
            if st != vocals_idx {
                left += acc[st][i][0];
                right += acc[st][i][1];
            }
        }
        instrumental.push(left);
        instrumental.push(right);
    }

    emit_split_progress(SplitProgress::Writing {
        stem: "instrumental".to_string(),
        done: n,
        total: n,
        percent: 100.0,
    });

    let instrumental_data = AudioData {
        samples: instrumental,
        sample_rate,
        channels: 2,
    };
    let instrumental_tmp = tmp_dir.join("instrumental.wav");
    write_audio(instrumental_tmp.to_str().unwrap(), &instrumental_data)?;

    emit_split_progress(SplitProgress::Stage("finalize"));

    let file_stem = Path::new(input_path)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("output");
    let base = PathBuf::from(&opts.output_dir).join(file_stem);

    let vocals_out = copy_to(
        vocals_tmp.to_str().unwrap(),
        &format!("{}_vocals.wav", base.to_string_lossy()),
    )?;
    let instrumental_out = copy_to(
        instrumental_tmp.to_str().unwrap(),
        &format!("{}_instrumental.wav", base.to_string_lossy()),
    )?;

    emit_split_progress(SplitProgress::Finished);

    Ok(VocalRemovalResult {
        instrumental_path: instrumental_out,
        vocals_path: vocals_out,
    })
}

fn copy_to(src: &str, dst: &str) -> Result<String> {
    fs::copy(src, dst)?;
    Ok(dst.to_string())
}
