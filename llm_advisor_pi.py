# src/llm_advisor.py - Complete working version with proper LLM advice
import requests
import numpy as np
import json
import time
import re
from datetime import datetime

class LLMAdvisor:
    def __init__(self, model_name="phi:2.7b-chat-v2-q2_K", temperature=0.3, timeout=30, model=None):
        # Allow both 'model_name' and 'model' parameters
        if model is not None:
            model_name = model
        
        self.model = model_name
        self.temperature = temperature
        self.timeout = timeout
        self.url = "http://localhost:11434"
        self.available = self._check_connection()
        
        self.advice_count = 0
        self.total_response_time = 0
        self.cache = {}
        self.last_advice = None
        
        if self.available:
            self._warm_model()
    
    def _check_connection(self):
        try:
            response = requests.get(f"{self.url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get('models', [])
                model_names = [m['name'] for m in models]
                print(f"✅ Connected to Ollama")
                print(f"  Available models: {model_names}")
                if self.model in model_names:
                    print(f"  ✅ Using model: {self.model}")
                    return True
                elif model_names:
                    self.model = model_names[0]
                    print(f"  ⚠️ Using fallback model: {self.model}")
                    return True
            return False
        except Exception as e:
            print(f"⚠️ Cannot connect to Ollama: {e}")
            return False
    
    def _warm_model(self):
        """Warm up the model with a simple prompt"""
        try:
            print(f"Warming up {self.model}...")
            response = requests.post(
                f"{self.url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": "You are an AI advisor. Respond with: Focus: stability, Epsilon: 0.0, Strategy: current, Confidence: 0.5, Suggestion: none",
                    "stream": False,
                    "options": {
                        "num_predict": 50,
                        "temperature": 0.1
                    }
                },
                timeout=30
            )
            if response.status_code == 200:
                print(f"✅ Model warmed up")
        except Exception as e:
            print(f"⚠️ Warmup error: {e}")
    
    def get_advice(self, context):
        """Get advice from LLM based on current context"""
        if not self.available:
            return self._default_advice()
        
        # Create cache key
        cache_key = f"{context.get('accuracy',0):.3f}_{context.get('epsilon',0):.3f}_{context.get('queries_used',0)}"
        
        if cache_key in self.cache:
            advice = self.cache[cache_key].copy()
            advice['from_cache'] = True
            return advice
        
        # Create detailed prompt for proper LLM advice
        prompt = self._create_detailed_prompt(context)
        
        try:
            start_time = time.time()
            
            # Call Ollama API
            response = requests.post(
                f"{self.url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "temperature": self.temperature,
                    "options": {
                        "num_predict": 200,
                        "temperature": self.temperature,
                        "top_k": 40,
                        "top_p": 0.9,
                        "repeat_penalty": 1.1
                    }
                },
                timeout=self.timeout
            )
            
            elapsed = time.time() - start_time
            self.total_response_time += elapsed
            
            if response.status_code == 200:
                result = response.json()
                text = result.get('response', '')
                
                # Parse the response
                advice = self._parse_llm_response(text)
                
                if advice:
                    advice['response_time'] = round(elapsed, 2)
                    self.advice_count += 1
                    self.last_advice = advice
                    self.cache[cache_key] = advice
                    self._print_advice(advice, elapsed)
                    return advice
            
            # If parsing failed, return default
            return self._default_advice()
            
        except requests.exceptions.Timeout:
            print(f"⚠️ LLM timeout after {self.timeout}s")
            return self._default_advice()
        except Exception as e:
            print(f"⚠️ LLM error: {e}")
            return self._default_advice()
    
    def _create_detailed_prompt(self, context):
        """Create a detailed prompt that forces proper LLM advice"""
        accuracy = context.get('accuracy', 0) * 100
        precision = context.get('precision', 0) * 100
        recall = context.get('recall', 0) * 100
        f1 = context.get('f1_score', 0)
        epsilon = context.get('epsilon', 1.0)
        queries_used = context.get('queries_used', 0)
        budget = context.get('budget', 50)
        pseudo_labels = context.get('pseudo_labels', 0)
        alert_rate = context.get('alert_rate', 0) * 100
        avg_reward = context.get('avg_reward', 0)
        steps = context.get('steps', 0)
        vae_score = context.get('vae_score', 0)
        
        prompt = f"""You are an expert AI advisor for IoT anomaly detection with active learning.

SYSTEM STATUS:
- Accuracy: {accuracy:.1f}%
- Precision: {precision:.1f}%
- Recall: {recall:.1f}%
- F1 Score: {f1:.3f}
- Exploration Rate (epsilon): {epsilon:.3f}
- Alert Rate: {alert_rate:.1f}%
- Average Reward: {avg_reward:.2f}
- Training Steps: {steps}
- VAE Anomaly Score: {vae_score:.3f}

ACTIVE LEARNING:
- Queries Used: {queries_used}/{budget}
- Pseudo-labels Generated: {pseudo_labels}

Based on this data, provide your expert advice in EXACTLY this JSON format (no extra text, no explanations, just pure JSON):

{{
    "analysis": "Your one-sentence analysis of the current situation",
    "epsilon_adjustment": X.XXX,
    "focus": "exploration OR exploitation OR active_learning OR stability",
    "al_strategy": "more_queries OR more_propagation OR balance OR current",
    "confidence": X.XXX,
    "suggestion": "Your specific actionable suggestion"
}}

RULES for epsilon_adjustment:
- If accuracy < 70% and epsilon < 0.3: set epsilon_adjustment to +0.05
- If accuracy > 85% and epsilon > 0.1: set epsilon_adjustment to -0.03
- If F1 score < 0.6: set epsilon_adjustment to +0.02
- If VAE score > 0.5: set epsilon_adjustment to +0.01
- Otherwise: set epsilon_adjustment to 0.00

Now respond with ONLY the JSON object, no other text:"""
        
        return prompt
    
    def _parse_llm_response(self, text):
        """Parse JSON response from LLM"""
        try:
            # Clean the text - remove any markdown or extra spaces
            text = text.strip()
            
            # Remove markdown code blocks if present
            text = re.sub(r'```json\s*', '', text)
            text = re.sub(r'```\s*', '', text)
            
            # Find JSON object
            json_match = re.search(r'\{[^{}]*\{[^{}]*\}[^{}]*\}|\{[^{}]*\}', text, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                advice = json.loads(json_str)
                
                # Validate and ensure all fields exist
                required_fields = ['analysis', 'epsilon_adjustment', 'focus', 'al_strategy', 'confidence', 'suggestion']
                for field in required_fields:
                    if field not in advice:
                        advice[field] = self._default_advice()[field]
                
                # Ensure numeric values are proper
                advice['epsilon_adjustment'] = float(np.clip(float(advice['epsilon_adjustment']), -0.1, 0.1))
                advice['confidence'] = float(np.clip(float(advice['confidence']), 0.0, 1.0))
                
                # Ensure focus is valid
                valid_focus = ['exploration', 'exploitation', 'active_learning', 'stability']
                if advice['focus'] not in valid_focus:
                    # Try to map common variations
                    focus_lower = advice['focus'].lower()
                    if 'explor' in focus_lower:
                        advice['focus'] = 'exploration'
                    elif 'exploit' in focus_lower:
                        advice['focus'] = 'exploitation'
                    elif 'active' in focus_lower or 'learn' in focus_lower:
                        advice['focus'] = 'active_learning'
                    else:
                        advice['focus'] = 'stability'
                
                # Ensure al_strategy is valid
                valid_strategies = ['more_queries', 'more_propagation', 'balance', 'current']
                if advice['al_strategy'] not in valid_strategies:
                    strat_lower = advice['al_strategy'].lower()
                    if 'query' in strat_lower:
                        advice['al_strategy'] = 'more_queries'
                    elif 'propagat' in strat_lower:
                        advice['al_strategy'] = 'more_propagation'
                    elif 'balance' in strat_lower:
                        advice['al_strategy'] = 'balance'
                    else:
                        advice['al_strategy'] = 'current'
                
                return advice
                
        except json.JSONDecodeError as e:
            print(f"JSON parse error: {e}")
            return None
        except Exception as e:
            print(f"Parse error: {e}")
            return None
    
    def _default_advice(self):
        """Return default advice when LLM unavailable"""
        return {
            'analysis': 'Continue current training strategy',
            'epsilon_adjustment': 0.0,
            'focus': 'stability',
            'al_strategy': 'current',
            'confidence': 0.5,
            'suggestion': 'Maintain current approach and monitor metrics'
        }
    
    def _print_advice(self, advice, elapsed):
        """Print formatted advice"""
        print(f"\n{'🤖'*30}")
        print(f"🤖 LLM ADVICE #{self.advice_count} (Phi-2)")
        print(f"{'🤖'*30}")
        print(f"⏱  Response: {elapsed:.1f}s")
        print(f"📊 Analysis: {advice['analysis']}")
        print(f"📈 Epsilon Adjustment: {advice['epsilon_adjustment']:+.3f}")
        print(f"🎯 Focus: {advice['focus']}")
        print(f"🎲 AL Strategy: {advice['al_strategy']}")
        print(f"✨ Confidence: {advice['confidence']:.1%}")
        print(f"💡 Suggestion: {advice['suggestion']}")
        if advice.get('from_cache'):
            print(f"📦 (from cache)")
        print(f"{'🤖'*30}\n")
    
    def get_stats(self):
        """Get advisor statistics"""
        return {
            'advice_count': self.advice_count,
            'cache_size': len(self.cache),
            'avg_response_time': self.total_response_time / max(1, self.advice_count),
            'model': self.model,
            'available': self.available
        }


class RewardShaper:
    def __init__(self, advisor, shaping_strength=0.3):
        self.advisor = advisor
        self.shaping_strength = shaping_strength
        self.shaping_history = []
    
    def shape(self, base_reward, state, action, context):
        """Shape reward based on LLM advice"""
        advice = self.advisor.get_advice(context)
        
        if not advice or advice['confidence'] < 0.3:
            return base_reward
        
        shaping_bonus = 0
        
        if advice['focus'] == 'exploration':
            novelty = np.abs(state[0] - 0.5) + np.abs(state[1] - 0.5) if len(state) > 1 else 0.5
            shaping_bonus += novelty * 0.5
        elif advice['focus'] == 'exploitation':
            shaping_bonus += 0.2
        elif advice['focus'] == 'active_learning':
            if action == 1:
                shaping_bonus += 0.3
        
        shaped_reward = base_reward + shaping_bonus * self.shaping_strength * advice['confidence']
        
        self.shaping_history.append({
            'base': base_reward,
            'shaped': shaped_reward,
            'bonus': shaping_bonus,
            'confidence': advice['confidence'],
            'focus': advice['focus']
        })
        
        if len(self.shaping_history) > 1000:
            self.shaping_history = self.shaping_history[-1000:]
        
        return shaped_reward


class ALAdvisor:
    def __init__(self, advisor):
        self.advisor = advisor
        self.strategy_history = []
    
    def advise_strategy(self, al_stats):
        """Get active learning strategy advice"""
        context = {
            'queries_used': al_stats.get('queries', 0),
            'budget': al_stats.get('budget', 50),
            'pseudo_labels': al_stats.get('pseudo_labels', 0),
            'propagations': al_stats.get('propagated', 0),
            'accuracy': al_stats.get('accuracy', 0),
            'epsilon': al_stats.get('epsilon', 0.1),
            'f1_score': al_stats.get('f1_score', 0),
            'avg_reward': al_stats.get('avg_reward', 0)
        }
        
        advice = self.advisor.get_advice(context)
        
        if advice and 'al_strategy' in advice:
            strategy = advice['al_strategy']
            confidence = advice['confidence']
            
            self.strategy_history.append({
                'timestamp': datetime.now(),
                'strategy': strategy,
                'confidence': confidence,
                'stats': al_stats
            })
            
            return strategy, confidence
        
        return 'balance', 0.5