#!/bin/bash

source_dir="/home/tiny_ling/projects/my_mistral/"
target_dir="/home/tiny_ling/projects/my_mistral/audios"
extension="mp3"

mkdir -p "$target_dir"

mv "$source_dir"/*."$extension" "$target_dir"

echo "All files with extension $extension have been moved to $target_dir"
