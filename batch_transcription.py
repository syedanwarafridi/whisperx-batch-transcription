import os
import json
import torch
import argparse
import concurrent.futures
from tqdm import tqdm
import whisperx
from pathlib import Path


def transcribe_audio_batch(audio_files, model, batch_size=8, device="cuda", language="en"):
    """
    Process a batch of audio files using WhisperX
    """
    results = []
    
    # Process audio files in mini-batches to optimize GPU memory usage
    for i in range(0, len(audio_files), batch_size):
        batch = audio_files[i:i + batch_size]
        print(f"Processing batch {i//batch_size + 1}/{(len(audio_files) + batch_size - 1)//batch_size}, files {i+1}-{min(i+batch_size, len(audio_files))}")
        
        # Load and transcribe audio files in current mini-batch
        batch_results = []
        for audio_file in batch:
            try:                

                # Load audio file
                audio = whisperx.load_audio(audio_file)
                # Transcribe with WhisperX
                result = model.transcribe(audio, language=language, batch_size=8)  
                
                # Store result with filename
                batch_results.append({
                    "filename": os.path.basename(audio_file),
                    "transcription": result
                })
                
            except Exception as e:
                print(f"Error processing {audio_file}: {str(e)}")
                batch_results.append({
                    "filename": os.path.basename(audio_file),
                    "error": str(e)
                })
        
        results.extend(batch_results)
        
        # Clear CUDA cache to prevent memory leaks
        if device == "cuda":
            torch.cuda.empty_cache()
    
    return results


def save_results(results, output_dir):
    """
    Save transcription results to files
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save individual transcription files
    for result in results:
        filename = result["filename"]
        base_name = os.path.splitext(filename)[0]
        
        # Save as JSON
        json_path = os.path.join(output_dir, f"{base_name}.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            # If there was an error, save the error message
            if "error" in result:
                json.dump({"error": result["error"]}, f, ensure_ascii=False, indent=2)
            else:
                json.dump(result["transcription"], f, ensure_ascii=False, indent=2)
        
        # Save plain text transcript if available
        if "transcription" in result and "segments" in result["transcription"]:
            text_path = os.path.join(output_dir, f"{base_name}.txt")
            with open(text_path, 'w', encoding='utf-8') as f:
                full_text = " ".join([segment.get("text", "") for segment in result["transcription"]["segments"]])
                f.write(full_text)


def main():
    parser = argparse.ArgumentParser(description="Batch transcribe WAV files using WhisperX")
    parser.add_argument("--input_dir", type=str, default="recordings", help="Directory containing WAV files")
    parser.add_argument("--output_dir", type=str, default="transcripts", help="Directory to save transcription results")
    parser.add_argument("--model_size", type=str, default="large-v2", help="WhisperX model size (tiny, base, small, medium, large-v1, large-v2)")
    parser.add_argument("--batch_size", type=int, default=8, help="Number of files to process in each GPU batch")
    parser.add_argument("--language", type=str, default="en", help="Language code for transcription")
    parser.add_argument("--compute_type", type=str, default="float16", help="Compute type (float16, float32, int8)")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use (cuda, cpu)")
    args = parser.parse_args()
    
    # Check if CUDA is available when device is set to cuda
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA is not available, falling back to CPU")
        args.device = "cpu"
        args.compute_type = "float32"
    
    # Get all WAV files in the input directory
    input_path = Path(args.input_dir)
    audio_files = list(str(p) for p in input_path.glob("**/*.wav") if p.is_file())

    
    if not audio_files:
        print(f"No WAV files found in {args.input_dir}")
        return
    
    print(f"Found {len(audio_files)} WAV files")
    
    # Load WhisperX model
    print(f"Loading WhisperX model: {args.model_size}")
    model = whisperx.load_model(
        args.model_size,
        device=args.device,
        compute_type=args.compute_type,
        language=args.language
    )
    
    # Process files in batches to manage memory
    print(f"Starting transcription with batch size: {args.batch_size}")
    results = transcribe_audio_batch(
        audio_files,
        model=model,
        batch_size=args.batch_size,
        device=args.device,
        language=args.language,
    )
    
    # Save results
    print(f"Saving transcriptions to {args.output_dir}")
    save_results(results, args.output_dir)
    
    print(f"Transcription complete. Processed {len(audio_files)} files.")

if __name__ == "__main__":
    main()
