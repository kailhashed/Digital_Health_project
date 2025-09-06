# Contributing to Audio Emotion Recognition System

Thank you for your interest in contributing to this project! This document provides guidelines for contributing to the Audio Emotion Recognition System.

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- Git
- CUDA-capable GPU (recommended)

### Development Setup

1. **Fork the repository**
   ```bash
   git clone https://github.com/your-username/Digital_Health_project.git
   cd Digital_Health_project
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-dev.txt  # Development dependencies
   ```

4. **Verify installation**
   ```bash
   python -c "import torch; print(f'PyTorch: {torch.__version__}')"
   python -c "import librosa; print(f'Librosa: {librosa.__version__}')"
   ```

## 📝 Code Style Guidelines

### Python Code Style

- Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) style guidelines
- Use [Black](https://black.readthedocs.io/) for code formatting
- Use [isort](https://isort.readthedocs.io/) for import sorting
- Use type hints where appropriate

### Documentation Style

- Use docstrings for all functions, classes, and modules
- Follow [Google docstring format](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings)
- Include parameter types and return values
- Provide usage examples for complex functions

### Example Function Documentation

```python
def extract_mel_spectrogram(audio_path: str, n_mels: int = 64) -> np.ndarray:
    """Extract mel-spectrogram features from audio file.
    
    Args:
        audio_path: Path to the audio file
        n_mels: Number of mel frequency bands to extract
        
    Returns:
        Mel-spectrogram as numpy array of shape (n_mels, time_steps)
        
    Raises:
        FileNotFoundError: If audio file doesn't exist
        ValueError: If audio file is corrupted or invalid
        
    Example:
        >>> features = extract_mel_spectrogram("audio.wav", n_mels=64)
        >>> print(features.shape)
        (64, 94)
    """
    # Implementation here
```

## 🏗️ Project Structure

### Directory Organization

```
src/
├── models/          # Model architectures
├── data/           # Data handling utilities
├── training/       # Training components
├── evaluation/     # Evaluation and metrics
└── utils/          # General utilities

scripts/            # Execution scripts
tests/              # Unit and integration tests
docs/               # Documentation files
```

### Adding New Models

1. **Create model class** in `src/models/`
2. **Add to model registry** in `src/models/__init__.py`
3. **Create trainer** if needed in `src/training/`
4. **Add tests** in `tests/models/`
5. **Update documentation**

## 🧪 Testing Guidelines

### Test Structure

```
tests/
├── unit/           # Unit tests
├── integration/    # Integration tests
├── fixtures/       # Test data and fixtures
└── conftest.py     # Pytest configuration
```

### Writing Tests

- Use `pytest` for testing framework
- Aim for >80% code coverage
- Include both positive and negative test cases
- Use descriptive test names

### Example Test

```python
import pytest
import torch
from src.models.custom_models import EmotionResNet

class TestEmotionResNet:
    """Test suite for EmotionResNet model."""
    
    def test_model_initialization(self):
        """Test that model initializes correctly."""
        model = EmotionResNet(num_classes=8)
        assert isinstance(model, torch.nn.Module)
        assert model.num_classes == 8
    
    def test_forward_pass(self):
        """Test forward pass with dummy input."""
        model = EmotionResNet(num_classes=8)
        dummy_input = torch.randn(2, 1, 64, 94)  # Batch, Channel, Height, Width
        
        output = model(dummy_input)
        
        assert output.shape == (2, 8)  # Batch size, num_classes
        assert torch.allclose(torch.sum(torch.softmax(output, dim=1), dim=1), 
                             torch.ones(2))  # Probabilities sum to 1
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/unit/test_models.py

# Run with verbose output
pytest -v
```

## 📊 Adding New Features

### Model Implementation

1. **Define the architecture** in appropriate model file
2. **Implement forward pass** with proper tensor shapes
3. **Add initialization method** with configurable parameters
4. **Include parameter counting** method
5. **Add comprehensive tests**

### Data Processing

1. **Create processing function** in `src/data/preprocessing.py`
2. **Handle edge cases** (empty files, corrupted audio, etc.)
3. **Add input validation** and error handling
4. **Document expected input/output formats**
5. **Include usage examples**

### Training Components

1. **Extend base trainer** or create new trainer class
2. **Implement training loop** with proper logging
3. **Add validation and early stopping**
4. **Include model checkpointing**
5. **Add progress tracking**

## 🐛 Bug Reports

### Before Submitting

1. **Search existing issues** to avoid duplicates
2. **Test with latest version** of the codebase
3. **Reproduce the bug** with minimal example
4. **Check if it's environment-specific**

### Bug Report Template

```markdown
**Bug Description**
A clear description of what the bug is.

**To Reproduce**
Steps to reproduce the behavior:
1. Go to '...'
2. Click on '....'
3. Scroll down to '....'
4. See error

**Expected Behavior**
A clear description of what you expected to happen.

**Environment**
- OS: [e.g. Ubuntu 20.04]
- Python version: [e.g. 3.9.7]
- PyTorch version: [e.g. 2.0.1]
- CUDA version: [e.g. 11.8]

**Additional Context**
Add any other context about the problem here.
```

## 💡 Feature Requests

### Feature Request Template

```markdown
**Feature Description**
A clear description of the feature you'd like to see.

**Use Case**
Describe the use case and why this feature would be valuable.

**Proposed Implementation**
If you have ideas about how to implement this feature.

**Alternatives Considered**
Alternative solutions or features you've considered.
```

## 📋 Pull Request Process

### Before Submitting

1. **Create feature branch** from main
2. **Write comprehensive tests** for new functionality
3. **Update documentation** as needed
4. **Run full test suite** and ensure all tests pass
5. **Check code style** with linting tools

### Pull Request Template

```markdown
**Description**
Brief description of changes made.

**Type of Change**
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

**Testing**
- [ ] Tests pass
- [ ] New tests added
- [ ] Manual testing completed

**Checklist**
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] No breaking changes (or documented)
```

### Review Process

1. **Automated checks** must pass (CI/CD)
2. **Code review** by maintainers
3. **Testing** on different environments
4. **Documentation review**
5. **Final approval** and merge

## 🔄 Development Workflow

### Git Workflow

1. **Create feature branch**
   ```bash
   git checkout -b feature/new-model-architecture
   ```

2. **Make changes and commit**
   ```bash
   git add .
   git commit -m "feat: add transformer model with attention"
   ```

3. **Keep branch updated**
   ```bash
   git fetch origin
   git rebase origin/main
   ```

4. **Push and create PR**
   ```bash
   git push origin feature/new-model-architecture
   ```

### Commit Message Format

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation only
- `style`: Code style changes
- `refactor`: Code refactoring
- `test`: Adding tests
- `chore`: Maintenance tasks

Examples:
```
feat(models): add attention mechanism to transformer
fix(data): handle empty audio files gracefully
docs(readme): update installation instructions
```

## 📚 Documentation Guidelines

### Code Documentation

- Document all public APIs
- Include usage examples
- Explain complex algorithms
- Document performance characteristics

### README Updates

- Keep installation instructions current
- Update performance benchmarks
- Add new model architectures
- Include new usage examples

### API Documentation

- Use Sphinx for API documentation generation
- Include parameter descriptions
- Provide return value specifications
- Add cross-references between related functions

## 🎯 Areas for Contribution

### High Priority

1. **Model Improvements**
   - New architectures (Vision Transformers, EfficientNet)
   - Attention mechanisms
   - Multi-modal approaches

2. **Data Augmentation**
   - Audio augmentation techniques
   - Synthetic data generation
   - Cross-dataset training

3. **Performance Optimization**
   - Model quantization
   - TensorRT optimization
   - Distributed training

### Medium Priority

1. **Evaluation Metrics**
   - Additional evaluation metrics
   - Cross-validation frameworks
   - Statistical significance testing

2. **Visualization**
   - Training progress visualization
   - Model interpretation tools
   - Feature visualization

3. **Deployment**
   - REST API implementation
   - Docker containerization
   - Cloud deployment guides

### Low Priority

1. **Documentation**
   - Tutorial notebooks
   - Video tutorials
   - Best practices guide

2. **Utilities**
   - Data conversion tools
   - Benchmarking scripts
   - Configuration management

## 🏆 Recognition

Contributors will be recognized in:

- **README.md** contributors section
- **CHANGELOG.md** for significant contributions
- **GitHub releases** acknowledgments
- **Academic publications** (if applicable)

## 📞 Getting Help

- **GitHub Issues**: Technical problems and bugs
- **GitHub Discussions**: General questions and ideas
- **Email**: Direct contact for sensitive issues

## 📜 Code of Conduct

This project follows the [Contributor Covenant Code of Conduct](https://www.contributor-covenant.org/version/2/1/code_of_conduct/). By participating, you agree to abide by its terms.

## 📄 License

By contributing to this project, you agree that your contributions will be licensed under the same MIT License that covers the project.

---

Thank you for contributing to the Audio Emotion Recognition System! 🎉
