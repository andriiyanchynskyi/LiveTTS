<template>
  <main style="max-width: 1000px; margin: 24px auto; font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; ">
    <h1 class="app-title">LiveTTS</h1>

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
          <select v-model="selectedLanguage" style="padding: 12px; border: 1px solid #ccc; border-radius: 4px; width: 55%; font-size: 16px; text-align: center;  font-style: italic;" :disabled="!selectedVoice">
            <option value="">{{ selectedVoice ? 'Language' : 'Set voice' }}</option>
            <option v-for="lang in availableLanguagesForVoice" :key="lang" :value="lang">
              {{ getLanguageDisplayName(lang) }}
            </option>
          </select>
        </div>
        <div style="flex: 1; min-width: 200px;">
          <label style="display: block; font-weight: 600; margin-bottom: 6px;">Select Voice</label>
          <select v-model="selectedVoice" style="padding: 12px; border: 1px solid #ccc; border-radius: 4px; width: 55%; font-size: 16px;  font-style: italic;">
            <option value="">Not Selected</option>
            <option v-for="voice in voices" :key="voice.name" :value="voice">
              {{ voice.name }} (Original: {{ getLanguageDisplayName(voice.language) }})
            </option>
          </select>
        </div>

        <div style="flex: 1; min-width: 200px;">
          <label style="display: block; font-weight: 600; margin-bottom: 6px;">
            Playback Speed: {{ speed.toFixed(1) }}x
          </label>
          <div style="display: flex; align-items: center; gap: 8px;">
            <span style="font-size: 12px; color: #666;">0.5x</span>
            <input 
              type="range" 
              v-model.number="speed" 
              min="0.5" 
              max="2.0" 
              step="0.1" 
              style="flex: 1; height: 6px; border-radius: 3px; outline: none;"
            />
            <span style="font-size: 12px; color: #666;">2.0x</span>
          </div>
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
           class="download-btn">
          Download
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
            Supported Languages:
{{
  ((voice.supported_languages && voice.supported_languages.length ? voice.supported_languages
    : (voice.m_supported_languages || availableLanguages)))
    .map(lang => getLanguageDisplayName(lang))
    .join(', ')
}}
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
import './styles.css'

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
const availableLanguages = ref([])
const speed = ref(1.0)

// API Functions
async function loadLanguages() {
  try {
    const { data } = await axios.get(`${apiBase}/languages`)
    availableLanguages.value = Array.isArray(data.languages) && data.languages.length ? data.languages : ['en']
  } catch (e) {
    errorMessage.value = 'Failed to load languages: ' + (e.response?.data?.detail || e.message)
    availableLanguages.value = ['en']
  }
}

async function loadVoices() {
  try {
    const { data } = await axios.get(`${apiBase}/voices`)
    voices.value = data.voices || []
    // Only auto-select first voice if voices are actually available
    if (voices.value.length > 0 && !selectedVoice.value) {
      selectedVoice.value = voices.value[0]  // Select first voice, not the entire array
    }
    systemStatus.value = 'Ready'
  } catch (e) {
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
const availableLanguagesForVoice = computed(() => {
  if (!selectedVoice.value) return []
  if (selectedVoice.value.fixed_language) return [selectedVoice.value.fixed_language]
  const supported = selectedVoice.value.supported_languages?.length ? selectedVoice.value.supported_languages : (selectedVoice.value.m_supported_languages || [])
  return supported.length ? supported : (availableLanguages.value.length ? availableLanguages.value : ['en'])
})

const isReadyForSynthesis = computed(() => {
  return Boolean(
    selectedVoice.value &&
    selectedLanguage.value &&
    selectedLanguage.value.trim() !== '' &&
    availableLanguagesForVoice.value.includes(selectedLanguage.value) &&
    text.value.trim() !== ''
  )
})

// Watchers
watch(selectedVoice, (newVoice) => {
  selectedLanguage.value = ''
  if (newVoice) {
    nextTick(() => {
      const langs = availableLanguagesForVoice.value
      selectedLanguage.value = newVoice.fixed_language || (langs.length ? langs : 'en')
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
  return languageNames[langCode] || String(langCode).toUpperCase()
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
  
  loading.value = true
  audioUrl.value = ''
  filename.value = ''
  errorMessage.value = ''
  
  try {
    const payload = {
      text: text.value,
      voice_name: selectedVoice.value.name,
      language: selectedLanguage.value,
      speed: speed.value
    }
    
    const { data } = await axios.post(`${apiBase}/synthesize`, payload)
    
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
  await diagGpu()
  // Only auto-select if voices are available and none selected
  if (voices.value.length > 0 && !selectedVoice.value) {
    selectedVoice.value = voices.value[0]  // Select first voice
    // Auto-select language for the first voice
    if (!selectedLanguage.value) {
      const langs = availableLanguagesForVoice.value
      selectedLanguage.value = selectedVoice.value.fixed_language || (langs.length ? langs[0] : 'en')
    }
  }
})
</script>
