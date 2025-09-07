<template>
  <main style="max-width: 1000px; margin: 24px auto; font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif;">
    <h1>LiveTTS</h1>

    <!-- Text Synthesis Section -->
    <section style="margin: 16px 0; padding: 16px; border: 1px solid #ddd; border-radius: 12px;">
      <label style="display: block; font-weight: 600; margin-bottom: 8px;">Text to Synthesize</label>
      <textarea 
        v-model="text" 
        rows="7" 
        style="width: 97%; padding: 12px; font-size: 16px; border: 1px solid #ccc; border-radius: 8px; resize: vertical; min-height: 110px;"
        placeholder="Enter your text here in the selected language..."
      ></textarea>

      <div style="display: flex; gap: 16px; margin-top: 16px; align-items: end; flex-wrap: wrap;">
        <div style="flex: 1; min-width: 200px;">
          <label style="display: block; font-weight: 600; margin-bottom: 6px;">Select Language</label>
          <select v-model="selectedLanguage" style="padding: 12px; border: 1px solid #ccc; border-radius: 4px; width: 55%; font-size: 16px;" :disabled="!selectedVoice">
            <option value="">{{ selectedVoice ? 'Choose a language...' : 'Select a voice first' }}</option>
            <option v-for="lang in availableLanguagesForVoice" :key="lang" :value="lang">
              {{ getLanguageDisplayName(lang) }}
            </option>
          </select>
        </div>
        <div style="flex: 1; min-width: 200px;">
          <label style="display: block; font-weight: 600; margin-bottom: 6px;">Select Voice</label>
          <select v-model="selectedVoice" style="padding: 12px; border: 1px solid #ccc; border-radius: 4px; width: 55%; font-size: 16px;">
            <option value="">Choose a voice...</option>
            <option v-for="voice in voices" :key="voice.name" :value="voice">
              {{ voice.name }} (Original: {{ getLanguageDisplayName(voice.language) }})
            </option>
          </select>
        </div>

        <button 
          :disabled="loading || !isReadyForSynthesis" 
          @click="synthesize"
          class="synthesize-btn"
        >
          {{ loading ? "Generating..." : "Synthesize" }}
        </button>
      </div>

      <div v-if="errorMessage" style="margin-top: 12px; padding: 12px; border-radius: 4px; background-color: #f8d7da; color: #721c24; border: 1px solid #f5c6cb;">
        {{ errorMessage }}
      </div>
      
      
    </section>

    <!-- Result Section -->
    <section v-if="audioUrl" style="margin-top: 16px; padding: 16px; border: 1px solid #ddd; border-radius: 12px;">
      <h3>Generated Audio</h3>
      <div style="margin-bottom: 12px;">
        <strong>Voice Used:</strong> {{ lastUsedVoice }}
      </div>
      <audio :src="apiBase + audioUrl" controls style="width: 100%;"></audio>
      <div style="margin-top: 12px;">
        <a :href="apiBase + audioUrl" :download="filename" 
           style="padding: 8px 16px; background-color: #007bff; color: white; text-decoration: none; border-radius: 4px;">
          Download WAV
        </a>
      </div>
    </section>

    <!-- Available Voices Section -->
    <section style="margin-top: 24px; padding: 16px; border: 1px solid #ddd; border-radius: 12px;">
      <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 16px;">
        <h3>Available Voices ({{ voices.length }})</h3>
        <button @click="refreshVoices" :disabled="refreshing"
                class="refresh-btn">
          {{ refreshing ? "Refreshing..." : "Refresh Voices" }}
        </button>
      </div>
      
      <div v-if="voices.length === 0" style="text-align: center; padding: 40px; color: #666;">
        <p>No voices found. Please check the volumes/voices/ directory.</p>
      </div>
      
      <div v-else style="display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: 16px;">
        <div v-for="voice in voices" :key="voice.name" 
             :style="{
               padding: '16px',
               border: '1px solid #eee',
               borderRadius: '8px',
               backgroundColor: selectedVoice?.name === voice.name ? '#f6f6f6' : '#f8f9fa',
               cursor: 'pointer',
               transition: 'all 0.2s ease'
             }"
             @click="selectedVoice = voice">
          <h4 style="margin: 0 0 8px 0; color: #333;">{{ voice.name }}</h4>
          <p style="margin: 0 0 8px 0; color: #666; font-size: 14px;">
            Original Language: {{ getLanguageDisplayName(voice.language) }}
          </p>
          <p style="margin: 0 0 8px 0; color: #666; font-size: 14px;">
            Supported Languages: {{ (voice.model_supported_languages || availableLanguages).map(lang => getLanguageDisplayName(lang)).join(', ') }}
          </p>
          <p v-if="voice.fixed_language" style="margin: 0 0 8px 0; color: #dc3545; font-size: 12px;">
            ⚠️ Fixed Language: {{ getLanguageDisplayName(voice.fixed_language) }}
          </p>
          <p style="margin: 0; font-size: 12px; color: #888;">
            Status: {{ voice.available ? 'Available' : 'Unavailable' }}
          </p>
          <div v-if="selectedVoice?.name === voice.name" class="selected-badge">
            Selected
          </div>
        </div>
      </div>
    </section>

    <!-- System Information -->
    <section style="margin-top: 24px; padding: 16px; border: 1px solid #ddd; border-radius: 12px; background-color: #f8f9fa;">
      <h3>System Information</h3>
      <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 16px; margin-top: 12px;">
        <div>
          <strong>Model:</strong> XTTS v2
        </div>
        <div>
          <strong>Languages:</strong> {{ availableLanguages.length }} Supported
        </div>
        <div>
          <strong>Voices Loaded:</strong> {{ voices.length }}
        </div>
        <div>
          <strong>Status:</strong> {{ systemStatus }}
        </div>
      </div>
    </section>

    <!-- GPU Diagnostics -->
    <section style="margin-top: 24px; color: #666;">
      <details>
        <summary>GPU Diagnostics</summary>
        <pre style="background-color: #f8f9fa; padding: 12px; border-radius: 4px; overflow-x: auto;">{{ gpuInfo }}</pre>
      </details>
    </section>
  </main>
</template>

<script setup>
import axios from 'axios'
import { ref, onMounted, computed, watch, nextTick } from 'vue'

// API Configuration
const apiBase = import.meta.env.VITE_API_BASE || 'http://localhost:8000'

// Reactive State
const text = ref('Hello! You can select different voices from the available list.')
const voices = ref([])
const selectedLanguage = ref('')
const selectedVoice = ref(null)
const audioUrl = ref('')
const filename = ref('')
const lastUsedVoice = ref('')
const loading = ref(false)
const refreshing = ref(false)
const errorMessage = ref('')
const systemStatus = ref('Loading...')
const gpuInfo = ref({})
const debugInfo = ref(true)
const availableLanguages = ref([])

// API Functions
async function loadLanguages() {
  try {
    console.log('Loading languages from:', `${apiBase}/languages`)
    const { data } = await axios.get(`${apiBase}/languages`)
    console.log('Languages response:', data)
    availableLanguages.value = data.languages && data.languages.length > 0 ? data.languages : ['en']
    console.log('Loaded languages:', availableLanguages.value)
  } catch (e) {
    console.error('Failed to load languages:', e)
    errorMessage.value = 'Failed to load languages: ' + (e.response?.data?.detail || e.message)
    availableLanguages.value = ['en'] // Fallback to English
  }
}

async function loadVoices() {
  try {
    console.log('Loading voices from:', `${apiBase}/voices`)
    const { data } = await axios.get(`${apiBase}/voices`)
    console.log('Voices response:', data)
    voices.value = data.voices || []
    
    console.log('Loaded voices:', voices.value)
    
    if (voices.value.length > 0 && !selectedVoice.value) {
      selectedVoice.value = voices.value[0]
      console.log('Auto-selected voice:', selectedVoice.value)
    }
    
    systemStatus.value = 'Ready'
  } catch (e) {
    console.error('Failed to load voices:', e)
    errorMessage.value = 'Failed to load voices: ' + (e.response?.data?.detail || e.message)
    systemStatus.value = 'Error'
  }
}

async function refreshVoices() {
  refreshing.value = true
  errorMessage.value = ''
  
  try {
    await axios.post(`${apiBase}/voices/refresh`)
    await loadVoices()
  } catch (e) {
    errorMessage.value = 'Failed to refresh voices: ' + (e.response?.data?.detail || e.message)
  } finally {
    refreshing.value = false
  }
}

async function diagGpu() {
  try {
    const { data } = await axios.get(`${apiBase}/gpu`)
    Object.assign(gpuInfo.value, data)
  } catch (e) {
    Object.assign(gpuInfo.value, { error: String(e) })
  }
}

// Computed Properties
const filteredVoices = computed(() => {
  if (!selectedLanguage.value) return voices.value
  return voices.value.filter(voice => voice.language === selectedLanguage.value)
})

const availableLanguagesForVoice = computed(() => {
  if (!selectedVoice.value) return []
  
  // If the voice has a fixed language in config, only show that language
  if (selectedVoice.value.fixed_language) {
    return [selectedVoice.value.fixed_language]
  }
  
  // Use model_supported_languages if available, otherwise fallback to all available languages
  const supported = selectedVoice.value.model_supported_languages || []
  
  // If no languages are specified in the model, use all available languages
  if (supported.length === 0) {
    return availableLanguages.value.length > 0 ? availableLanguages.value : ['en']
  }
  
  return supported
})

const isReadyForSynthesis = computed(() => {
  return selectedVoice.value &&
         selectedLanguage.value &&
         selectedLanguage.value.trim() !== '' &&
         availableLanguagesForVoice.value.includes(selectedLanguage.value) &&
         text.value.trim() !== ''
})

// Watchers
watch(selectedVoice, (newVoice, oldVoice) => {
  console.log('Voice changed from:', oldVoice, 'to:', newVoice)
  
  // Reset selected language when voice changes
  selectedLanguage.value = ''
  
  if (newVoice) {
    // Wait for next tick to ensure availableLanguagesForVoice is updated
    nextTick(() => {
      const availableLangs = availableLanguagesForVoice.value
      if (availableLangs.length > 0) {
        // If voice has fixed language, use it; otherwise use first available
        selectedLanguage.value = newVoice.fixed_language || availableLangs[0]
        console.log('Auto-selected language for voice:', selectedLanguage.value)
      } else {
        // Fallback to 'en' if no languages are available
        selectedLanguage.value = 'en'
        console.log('No language available for voice, defaulting to English:', newVoice.name)
      }
    })
  }
})

// Helper Functions
function getLanguageDisplayName(langCode) {
  const languageNames = {
    'en': 'English',
    'es': 'Spanish',
    'fr': 'French',
    'de': 'German',
    'it': 'Italian',
    'pt': 'Portuguese',
    'pl': 'Polish',
    'tr': 'Turkish',
    'ru': 'Russian',
    'nl': 'Dutch',
    'cs': 'Czech',
    'ar': 'Arabic',
    'zh-cn': 'Chinese (Simplified)',
    'hu': 'Hungarian',
    'ko': 'Korean',
    'ja': 'Japanese',
    'hi': 'Hindi'
  }
  return languageNames[langCode] || langCode.toUpperCase()
}

async function synthesize() {
  if (!selectedVoice.value) {
    errorMessage.value = 'Please select a voice'
    return
  }
  
  if (!selectedLanguage.value || 
      selectedLanguage.value.trim() === '' || 
      !availableLanguagesForVoice.value.includes(selectedLanguage.value)) {
    errorMessage.value = 'Please select a valid language'
    return
  }
  
  console.log('Synthesizing with voice:', selectedVoice.value)
  console.log('Synthesizing with language:', selectedLanguage.value)
  console.log('Text:', text.value)
  
  loading.value = true
  audioUrl.value = ''
  filename.value = ''
  errorMessage.value = ''
  
  try {
    const payload = {
      text: text.value,
      voice_name: selectedVoice.value.name,
      language: selectedLanguage.value
    }
    
    console.log('Sending payload:', payload)
    console.log('Payload language type:', typeof payload.language)
    console.log('Payload language length:', payload.language ? payload.language.length : 'undefined')
    console.log('Payload language trimmed:', payload.language ? payload.language.trim() : 'undefined')
    
    const { data } = await axios.post(`${apiBase}/synthesize`, payload)
    console.log('Synthesis response:', data)
    
    audioUrl.value = data.audio_url
    filename.value = data.filename
    lastUsedVoice.value = data.voice_used
    
  } catch (e) {
    console.error('Synthesis error:', e)
    console.error('Error response:', e.response?.data)
    errorMessage.value = 'Synthesis failed: ' + (e.response?.data?.detail || e.message)
  } finally {
    loading.value = false
  }
}

// Initialize
onMounted(async () => {
  await loadLanguages()
  await loadVoices()
  diagGpu()
  
  // Ensure we have a voice selected and set initial language
  if (voices.value.length > 0 && !selectedVoice.value) {
    selectedVoice.value = voices.value[0]
    console.log('Auto-selected voice:', selectedVoice.value)
  }
  
  // Set initial language based on selected voice
  if (selectedVoice.value && !selectedLanguage.value) {
    const availableLangs = availableLanguagesForVoice.value
    if (availableLangs.length > 0) {
      selectedLanguage.value = selectedVoice.value.fixed_language || availableLangs[0]
      console.log('Auto-selected language for voice:', selectedLanguage.value)
    } else {
      selectedLanguage.value = 'en'
      console.log('No language available for voice, defaulting to English:', selectedVoice.value.name)
    }
  }
})
</script>

<style>
/* Minimalistic button styles with custom color */
.synthesize-btn {
  padding: 12px 20px;
  border: none;
  border-radius: 6px;
  background-color: rgb(246, 192, 80);
  color: white;
  font-size: 16px;
  font-weight: 500;
  cursor: pointer;
  transition: all 0.2s ease;
  min-width: 160px;
  height: 50px;
}

.synthesize-btn:hover:not(:disabled) {
  background-color: rgb(240, 180, 70);
  transform: translateY(-1px);
}

.synthesize-btn:disabled {
  background-color: #ccc;
  cursor: not-allowed;
  transform: none;
}

.refresh-btn {
  padding: 8px 16px;
  border: none;
  border-radius: 6px;
  background-color: rgb(246, 192, 80);
  color: white;
  font-size: 14px;
  font-weight: 500;
  cursor: pointer;
  transition: all 0.2s ease;
}

.refresh-btn:hover:not(:disabled) {
  background-color: rgb(240, 180, 70);
  transform: translateY(-1px);
}

.refresh-btn:disabled {
  background-color: #ccc;
  cursor: not-allowed;
  transform: none;
}

.selected-badge {
  margin-top: 8px;
  padding: 4px 8px;
  background-color: rgb(246, 192, 80);
  color: white;
  border-radius: 4px;
  font-size: 12px;
  display: inline-block;
  font-weight: 500;
}

/* Enhanced styling */
select, input, textarea {
  font-family: inherit;
}

select:disabled {
  background-color: #f8f9fa;
  color: #6c757d;
  cursor: not-allowed;
}

/* Responsive design */
@media (max-width: 768px) {
  main {
    margin: 12px;
    max-width: none;
  }
  
  .flex-wrap {
    flex-direction: column;
    align-items: stretch;
  }
  
  .grid {
    grid-template-columns: 1fr;
  }
  
  .synthesize-btn {
    width: 100%;
    margin-top: 12px;
  }
}

/* Voice selection animation */
.voice-card:hover {
  transform: translateY(-2px);
  box-shadow: 0 4px 8px rgba(0,0,0,0.1);
}
</style>
