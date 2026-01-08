# CLAUDE.md

# Stable Diffusion WebUI Extension Development - Cursor Rules

## Project Context
You are assisting a Stable Diffusion enthusiast who is building and modifying extensions for AUTOMATIC1111's WebUI. The user has a 12GB RTX 4070 setup with 64GB RAM, uses StabilityMatrix for package management, and wants to add custom features through extensions while maintaining a clean development workflow.

## Core Extension Development Rules

### 🚫 NEVER MODIFY SYSTEM FILES
- **FORBIDDEN**: Never modify modules/, scripts/, launch.py, webui.py, or any core WebUI files
- **ONLY WORK IN**: `extensions/` folder and new extension directories
- **RULE**: All functionality must be implemented through the extension system

### 📁 Extension Structure Requirements
```
extensions/your-extension-name/
├── install.py                 # Optional: dependency installer
├── scripts/
│   └── your_extension.py      # Main extension script
├── javascript/                # Optional: JS files for UI
├── localizations/             # Optional: translation files
├── style.css                  # Optional: custom CSS
└── preload.py                 # Optional: command line arg handler
```

### 🎯 Extension Patterns
When creating extensions, follow these patterns:

**Pattern 1: Custom Script**
```python
import modules.scripts as scripts
import gradio as gr
from modules.processing import process_images

class Script(scripts.Script):
    def title(self):
        return "Your Extension Name"
    
    def ui(self, is_img2img):
        # Define UI components
        return [components]
    
    def run(self, p, *args):
        # Extension logic
        return process_images(p)
```

**Pattern 2: UI Tab**
```python
import modules.scripts as scripts
import gradio as gr

def on_ui_tabs():
    with gr.Blocks() as interface:
        # Your UI here
        pass
    return [(interface, "Tab Name", "tab_id")]

scripts.script_callbacks.on_ui_tabs(on_ui_tabs)
```

### 🔧 Development Guidelines

**Gradio Integration**
- Use `gr.Textbox()`, `gr.Slider()`, `gr.Dropdown()` for inputs
- Use `gr.Image()` for image inputs/outputs
- Use `gr.Button().click()` for interactions
- Leverage `gr.Blocks()` for complex layouts

**Extension Safety**
- Use `scripts.basedir()` to get extension directory
- Import extension modules using sys.path extension
- Handle errors gracefully with try/catch blocks
- Validate user inputs before processing

**StabilityMatrix Compatibility**
- Ensure extensions work with shared model directories
- Use command line args for model paths when possible
- Test with `--ckpt-dir`, `--lora-dir`, `--vae-dir` configurations

## 🧪 Sandbox Development Setup

### Multiple WebUI Instances for Testing
Create separate testing instances to avoid breaking your main setup:

**Method 1: StabilityMatrix Multiple Installations**
- Install separate WebUI instances in StabilityMatrix
- Use shared model directories to save space
- Configure different ports: `--port 7861`, `--port 7862`

**Method 2: Command Line Instances**
```bash
# Testing instance with separate data directory
python launch.py --data-dir ./test-data --port 7861 --api

# Sandbox with CPU mode for quick testing
python launch.py --use-cpu all --precision full --no-half --port 7862
```

### Shared Model Configuration
In your webui-user.bat/sh:
```bash
set COMMANDLINE_ARGS=--ckpt-dir "D:\shared-models\Checkpoints" --lora-dir "D:\shared-models\Lora" --vae-dir "D:\shared-models\VAE" --embeddings-dir "D:\shared-models\Embeddings"
```

## 🔄 Converting Gradio Demos to Extensions

When converting standalone Gradio applications:

1. **Extract the core function**:
   ```python
   def your_processing_function(image, prompt, *args):
       # Core logic here
       return output_image
   ```

2. **Wrap in extension structure**:
   ```python
   class YourExtensionScript(scripts.Script):
       def ui(self, is_img2img):
           with gr.Group():
               input_image = gr.Image()
               prompt = gr.Textbox()
               process_btn = gr.Button("Process")
           
           process_btn.click(
               fn=your_processing_function,
               inputs=[input_image, prompt],
               outputs=[output_image]
           )
   ```

3. **Handle dependencies in install.py**:
   ```python
   import launch
   
   if not launch.is_installed("your_package"):
       launch.run_pip("install your_package", "description")
   ```

## 🛠️ Development Best Practices

### Code Organization
- Keep extension logic in dedicated modules
- Use descriptive function and variable names
- Comment complex algorithms and UI interactions
- Follow Python PEP 8 style guidelines

### Error Handling
```python
try:
    result = process_function(inputs)
except Exception as e:
    print(f"Extension Error: {str(e)}")
    return None
```

### UI/UX Considerations
- Provide clear labels and descriptions
- Add tooltips for complex options
- Use appropriate input validation
- Show progress indicators for long operations

### Testing Workflow
1. **Development**: Work in sandbox WebUI instance
2. **Testing**: Test with various inputs and edge cases
3. **Integration**: Test with other extensions enabled
4. **Performance**: Monitor VRAM usage and generation times

## 📦 Extension Distribution

### File Organization
- Keep extension files minimal and focused
- Include clear README.md with installation instructions
- Provide example usage and screenshots
- Document any special requirements or dependencies

### Version Control
- Use git for extension development
- Tag stable releases
- Maintain changelog for updates
- Include license information

## 🔧 Hardware-Specific Optimizations

**For your RTX 4070 12GB setup:**
- Use `--xformers` for memory optimization
- Consider `--medvram` if needed for large extensions
- Test memory usage with different batch sizes
- Optimize for your specific VRAM limitations

**StabilityMatrix Integration:**
- Ensure extensions work with StabilityMatrix's package management
- Test with different WebUI versions supported by StabilityMatrix
- Use shared directories effectively to save disk space

## 🎨 UI Enhancement Features

When adding dropdowns and controls:
```python
# Dynamic dropdown based on available models
def get_available_models():
    return ["model1", "model2", "model3"]

model_dropdown = gr.Dropdown(
    choices=get_available_models(),
    label="Select Model",
    value=get_available_models()[0] if get_available_models() else None
)
```

## 🔍 Debugging and Troubleshooting

- Use print statements for debugging (they appear in console)
- Check browser developer tools for JavaScript errors
- Monitor VRAM usage during extension operation
- Test extension loading/unloading behavior
- Verify compatibility with core WebUI updates

Remember: The goal is to enhance WebUI functionality through safe, maintainable extensions that integrate smoothly with the existing ecosystem while providing powerful new capabilities for Stable Diffusion workflows.

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is the z-tipo-extension, a multi-platform AI extension that works with SD-WebUI, ComfyUI, and Forge. It utilizes TIPO (Text to Image with text Presampling for Prompt Optimization) and DanTagGen models to enhance text-to-image prompts by generating detailed tags and natural language descriptions.

## Dependencies and Installation

The extension has several key dependencies:
- `tipo-kgen>=0.1.9` - Core TIPO/KGen library for prompt generation
- `transformers` - For neural network models
- `llama-cpp-python` - For GGUF model inference (auto-installed)

The installation is handled by two key files:
- `install.py` - Handles automatic dependency installation for SD-WebUI/Forge
- `tipo_installer.py` - Contains installation logic and version management

## Architecture

### Core Components

#### 1. Node System (ComfyUI)
- **Location**: `nodes/tipo.py`
- **Classes**: `TIPO`, `TIPOOperation`, `TIPOFormat`
- **Purpose**: Provides ComfyUI nodes for prompt generation and formatting
- **Key Features**: Model loading, prompt processing, strength parsing, format application

#### 2. Script System (SD-WebUI/Forge)
- **Location**: `scripts/tipo.py`
- **Class**: `TIPOScript`
- **Purpose**: Integrates with SD-WebUI through the scripts system
- **Key Features**: UI components, gradio interface, prompt timing control

### Key Architecture Patterns

1. **Model Management**: Centralized model loading through `kgen.models`
2. **Prompt Processing Pipeline**: 
   - Parse input prompts for attention syntax `(tag:weight)` and `[tag:weight]`
   - Separate tags into categories (special, characters, copyrights, artist, general, etc.)
   - Generate enhanced prompts using TIPO models
   - Apply custom formatting templates
   - Handle prompt strength and BREAK tokens

3. **Multi-Platform Support**: 
   - ComfyUI: Node-based interface
   - SD-WebUI/Forge: Script-based integration with gradio UI

### Model System

- **Model Directory**: `models/` (auto-created)
- **Supported Formats**: Both HuggingFace models and GGUF quantized models
- **Auto-Download**: Models are downloaded on-demand when selected
- **Device Management**: Supports both CPU and CUDA inference

### Prompt Format System

The extension uses a template-based formatting system with placeholders:
- `<|special|>` - Character count tags (1girl, 1boy, etc.)
- `<|characters|>` - Character names
- `<|copyrights|>` - Series/franchise names  
- `<|artist|>` - Artist tags
- `<|general|>` - General descriptive tags
- `<|quality|>` - Quality tags (masterpiece, best quality, etc.)
- `<|meta|>` - Resolution/meta tags
- `<|rating|>` - Content rating tags
- `<|generated|>` - AI-generated natural language
- `<|extended|>` - Extended natural language descriptions

## Development Guidelines

### Code Organization
- Keep ComfyUI nodes in `nodes/` directory
- Keep SD-WebUI scripts in `scripts/` directory  
- Shared utilities can be imported from the KGen library
- Installation logic goes in `install.py` and `tipo_installer.py`

### Model Integration
- Always use the centralized model management from `kgen.models`
- Handle both HuggingFace and GGUF model formats
- Implement proper device management for CPU/CUDA switching
- Cache loaded models to avoid redundant loading

### Prompt Processing
- Use the existing attention syntax parsing functions
- Maintain compatibility with SD-WebUI's prompt attention system
- Handle BREAK tokens and strength modifiers properly
- Apply formatting templates consistently

### UI Considerations
- For SD-WebUI: Use gradio components and follow existing UI patterns
- For ComfyUI: Define proper INPUT_TYPES and RETURN_TYPES for nodes
- Maintain backward compatibility with existing workflows

## Common Development Tasks

### Adding New Models
1. Update model lists in the KGen library
2. Ensure download logic handles the new model format
3. Test loading with both CPU and CUDA devices

### Modifying Prompt Formats
1. Update format templates in the appropriate constants
2. Ensure placeholder handling works correctly
3. Test with both formatted and unformatted output modes

### Debugging Issues
- Check model loading logs for download/loading errors
- Verify prompt parsing is handling attention syntax correctly
- Test with different model formats (HuggingFace vs GGUF)
- Ensure device switching works properly between CPU/CUDA

## Platform-Specific Notes

### SD-WebUI/Forge Integration
- Extension loads through the scripts system
- UI elements are integrated into the generation interface
- Supports both "before" and "after" processing timing
- Handles batch processing and HR-fix scenarios

### ComfyUI Integration  
- Provides three main node types: TIPO, TIPOOperation, TIPOFormat
- Nodes can be chained for complex workflows
- Supports custom model selection and parameter tuning
- Returns multiple output formats (formatted, unformatted, etc.)

The codebase follows a modular design that allows the same core functionality to be exposed through different interfaces depending on the target platform.
