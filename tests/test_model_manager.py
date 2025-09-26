"""Tests for the model_manager module."""

import os
import tempfile
from pathlib import Path
from typing import Dict, Any
from unittest import TestCase
from unittest.mock import patch, mock_open, MagicMock

import pytest
import yaml

from assist.model_manager import (
    _load_custom_openai_config,
    select_chat_model,
    get_model_pair,
)


class TestCustomOpenAIConfig(TestCase):
    def setUp(self):
        # Change to a temporary directory for testing
        self.original_cwd = os.getcwd()
        self.temp_dir = tempfile.mkdtemp()
        os.chdir(self.temp_dir)

    def tearDown(self):
        os.chdir(self.original_cwd)

    def test_load_custom_openai_config_file_not_exists(self):
        """Test loading config when file doesn't exist."""
        result = _load_custom_openai_config()
        self.assertIsNone(result)

    def test_load_custom_openai_config_valid_file(self):
        """Test loading config from valid YAML file."""
        config_data = {
            'url': 'http://localhost:8000/v1',
            'model': 'custom-model',
            'api_key': 'test-key-123'
        }
        
        # Create config file
        with open('llm-config.yml', 'w') as f:
            yaml.dump(config_data, f)
        
        result = _load_custom_openai_config()
        self.assertEqual(result, config_data)

    def test_load_custom_openai_config_missing_fields(self):
        """Test loading config with missing required fields."""
        config_data = {
            'url': 'http://localhost:8000/v1',
            'model': 'custom-model'
            # Missing api_key
        }
        
        with open('llm-config.yml', 'w') as f:
            yaml.dump(config_data, f)
        
        result = _load_custom_openai_config()
        self.assertIsNone(result)

    def test_load_custom_openai_config_invalid_yaml(self):
        """Test loading config with invalid YAML."""
        with open('llm-config.yml', 'w') as f:
            f.write('invalid: yaml: content: [')
        
        result = _load_custom_openai_config()
        self.assertIsNone(result)

    def test_load_custom_openai_config_not_dict(self):
        """Test loading config when YAML is not a dictionary."""
        with open('llm-config.yml', 'w') as f:
            yaml.dump(['not', 'a', 'dict'], f)
        
        result = _load_custom_openai_config()
        self.assertIsNone(result)


class TestSelectChatModel(TestCase):
    def setUp(self):
        self.original_cwd = os.getcwd()
        self.temp_dir = tempfile.mkdtemp()
        os.chdir(self.temp_dir)

    def tearDown(self):
        os.chdir(self.original_cwd)

    @patch('assist.model_manager.ChatOpenAI')
    def test_select_chat_model_with_custom_config(self, mock_chat_openai):
        """Test select_chat_model uses custom configuration when available."""
        config_data = {
            'url': 'http://localhost:8000/v1',
            'model': 'custom-model',
            'api_key': 'test-key-123'
        }
        
        with open('llm-config.yml', 'w') as f:
            yaml.dump(config_data, f)
        
        mock_instance = MagicMock()
        mock_chat_openai.return_value = mock_instance
        
        result = select_chat_model('gpt-4o', 0.7)
        
        mock_chat_openai.assert_called_once_with(
            model='custom-model',
            temperature=0.7,
            api_key='test-key-123',
            base_url='http://localhost:8000/v1'
        )
        self.assertEqual(result, mock_instance)

    @patch('assist.model_manager.ChatOpenAI')
    def test_select_chat_model_fallback_gpt(self, mock_chat_openai):
        """Test select_chat_model falls back to default behavior for GPT models."""
        mock_instance = MagicMock()
        mock_chat_openai.return_value = mock_instance
        
        result = select_chat_model('gpt-4o', 0.5)
        
        mock_chat_openai.assert_called_once_with(model='gpt-4o', temperature=0.5)
        self.assertEqual(result, mock_instance)

    @patch('assist.model_manager.ChatOllama')
    def test_select_chat_model_fallback_ollama(self, mock_chat_ollama):
        """Test select_chat_model falls back to Ollama for non-GPT models."""
        mock_instance = MagicMock()
        mock_chat_ollama.return_value = mock_instance
        
        result = select_chat_model('llama2', 0.3)
        
        mock_chat_ollama.assert_called_once_with(model='llama2', temperature=0.3)
        self.assertEqual(result, mock_instance)


class TestGetModelPair(TestCase):
    def setUp(self):
        self.original_cwd = os.getcwd()
        self.temp_dir = tempfile.mkdtemp()
        os.chdir(self.temp_dir)

    def tearDown(self):
        os.chdir(self.original_cwd)

    @patch('assist.model_manager.select_chat_model')
    def test_get_model_pair_with_custom_config(self, mock_select):
        """Test get_model_pair uses custom config for both models."""
        config_data = {
            'url': 'http://localhost:8000/v1',
            'model': 'custom-model',
            'api_key': 'test-key-123'
        }
        
        with open('llm-config.yml', 'w') as f:
            yaml.dump(config_data, f)
        
        mock_model1 = MagicMock()
        mock_model2 = MagicMock()
        mock_select.side_effect = [mock_model1, mock_model2]
        
        plan_llm, exec_llm = get_model_pair('gpt-4o', 0.6)
        
        # Both calls should be made with the original model name
        self.assertEqual(mock_select.call_count, 2)
        mock_select.assert_any_call('gpt-4o', 0.6)
        self.assertEqual(plan_llm, mock_model1)
        self.assertEqual(exec_llm, mock_model2)

    @patch('assist.model_manager.select_chat_model')
    def test_get_model_pair_fallback_behavior(self, mock_select):
        """Test get_model_pair falls back to default behavior without custom config."""
        mock_model1 = MagicMock()
        mock_model2 = MagicMock()
        mock_select.side_effect = [mock_model1, mock_model2]
        
        plan_llm, exec_llm = get_model_pair('gpt-4o', 0.4)
        
        # Should be called twice - once for planning model, once for execution model (gpt-4o-mini)
        self.assertEqual(mock_select.call_count, 2)
        mock_select.assert_any_call('gpt-4o', 0.4)
        mock_select.assert_any_call('gpt-4o-mini', 0.4)
        self.assertEqual(plan_llm, mock_model1)
        self.assertEqual(exec_llm, mock_model2)