<template>
  <main class="container">
    <h1 class="app-title">LiveTTS</h1>

    <!-- Text Synthesis -->
    <section class="card">
      <label class="label">Text to Synthesize</label>
      <textarea
        v-model="text"
        rows="7"
        class="textarea"
        placeholder="Enter your text here in the selected language..."
      ></textarea>

      <div class="row wrap gap-16 align-end">
        <div class="col">
          <label class="label">Select Language</label>
          <select v-model="selectedLanguage" class="select center italic" :disabled="!selectedVoice">
            <option value="">{{ selectedVoice ? 'Language' : 'Set voice' }}</option>
            <option v-for="lang in availableLanguagesForVoice" :key="lang" :value="lang">
              {{ getLanguageDisplayName(lang) }}
            </option>
          </select>
        </div>

        <div class="col">
          <label class="label">Select Voice</label>
          <select v-model="selectedVoice" class="select italic">
            <option value="">Not Selected</option>
            <option v-for="voice in voices" :key="voice.name" :value="voice">
              {{ voice.name }}<span v-if="voice.is_zero_shot"> (zero-shot)</span>
              <template v-else> (Original: {{ getLanguageDisplayName(voice.language) }})</template>
            </option>
          </select>
        </div>

        <div class="col">
          <label class="label">Playback Speed: {{ speed.toFixed(1) }}x</label>
          <div class="range-row">
            <span class="muted small">0.5x</span>
            <input type="range" v-model.number="speed" min="0.5" max="2.0" step="0.1" class="range" />
            <span class="muted small">2.0x</span>
          </div>
        </div>

        <button :disabled="loading || !isReadyForSynthesis" @click="synthesize" class="btn synthesize-btn">
          {{ loading ? "Generating..." : "Synthesize" }}
        </button>
      </div>

      <div v-if="errorMessage" class="alert alert-error">
        {{ errorMessage }}
      </div>
    </section>

    <!-- Zero-shot uploader -->
    <section class="card">
      <div class="row justify-end">
        <div class="upload-icon-wrapper"
             @click="triggerFileDialog"
             @mouseenter="showTooltip = true"
             @mouseleave="showTooltip = false">
          <span class="upload-icon" title="Upload reference.wav">📎</span>
          <input ref="fileInput" type="file" accept="audio/wav,.wav" class="hidden" @change="handleFileSelected" />
          <div v-if="showTooltip" class="tooltip">
            Upload a clean mono WAV (16k–24kHz, 5–15 seconds, up to 20MB).
          </div>
        </div>
      </div>

      <div v-if="zeroShotActive" class="inline-info">
        <div><strong>Zero-shot voice ready:</strong> {{ zeroShotName }}</div>
        <button class="btn refresh-btn" @click="clearZeroShot" :disabled="loading">Clear</button>
      </div>
    </section>

    <!-- Result -->
    <section v-if="audioUrl" class="card">
      <h3>Generated Audio</h3>
      <div class="row gap-8">
        <div><strong>Voice Used:</strong> {{ lastUsedVoice }}</div>
      </div>
      <audio :src="apiBase + audioUrl" controls class="audio"></audio>
      <div class="spacer-12">
        <a :href="apiBase + audioUrl" :download="filename" class="btn download-btn">Download</a>
      </div>
    </section>

    <!-- Voices -->
    <section class="card">
      <div class="row space-between align-center">
        <h3>Available Voices ({{ voices.length }})</h3>
        <button @click="refreshVoices" :disabled="refreshing" class="btn refresh-btn">
          {{ refreshing ? "Refreshing..." : "Refresh" }}
        </button>
      </div>

      <div v-if="voices.length === 0" class="empty">
        <p>No voices found. Please check the /volumes/voices/xtts_v2/ directory.</p>
      </div>

      <div v-else class="grid">
        <div v-for="voice in voices" :key="voice.name"
             class="voice-card"
             :class="{ selected: selectedVoice?.name === voice.name }"
             @click="selectedVoice = voice">
          <h4 class="voice-title">
            {{ voice.name }}
            <span v-if="voice.is_zero_shot" class="badge">zero-shot</span>
          </h4>
          <p class="muted small">
            <template v-if="!voice.is_zero_shot">
              Original Language: {{ getLanguageDisplayName(voice.language) }}
            </template>
            <template v-else>
              Base Model: {{ voice.base_model || 'xtts_v2_base' }}
            </template>
          </p>
          <p class="muted small">
            Supported Languages:
            {{
              ((voice.supported_languages && voice.supported_languages.length ? voice.supported_languages
                : (voice.m_supported_languages || availableLanguages)))
                .map(lang => getLanguageDisplayName(lang))
                .join(', ')
            }}
          </p>
          <p v-if="voice.fixed_language" class="warn small">
            ⚠️ Fixed Language: {{ getLanguageDisplayName(voice.fixed_language) }}
          </p>
          <p class="muted xsmall">Status: {{ voice.available ? 'Available' : 'Unavailable' }}</p>
          <div v-if="selectedVoice?.name === voice.name" class="selected-badge">Selected</div>
        </div>
      </div>
    </section>

    <!-- System Info -->
    <section class="card muted-bg system-info">
      <h3>System Information</h3>
      <div class="sys-grid">
        <div><strong>Model:</strong> XTTS v2</div>
        <div><strong>Languages:</strong> {{ availableLanguages.length }} Supported</div>
        <div><strong>Voices Loaded:</strong> {{ voices.length }}</div>
        <div><strong>Status:</strong> {{ systemStatus }}</div>
      </div>
    </section>

    <!-- GPU Diagnostics -->
    <section class="section-inline muted">
      <details>
        <summary>GPU Diagnostics</summary>
        <pre class="code-block">{{ gpuInfo }}</pre>
      </details>
    </section>
  </main>
</template>

<script setup>



import axios from 'axios'
import { ref, onMounted, computed, watch, nextTick } from 'vue'
import './styles.css'

// Base API URL
const apiBase = import.meta.env.VITE_API_BASE || 'http://localhost:8000'

// Reactive state
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

// Zero-shot helpers
const fileInput = ref(null)
const showTooltip = ref(false)
const zeroShotActive = ref(false)
const zeroShotName = ref('')

// API calls
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
    // mark zero-shot presence
    zeroShotActive.value = voices.value.some(v => v.is_zero_shot)
    if (voices.value.length > 0 && !selectedVoice.value) {
      selectedVoice.value = voices.value[0]
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

// Computed helpers
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
      selectedLanguage.value = newVoice.fixed_language || (langs.length ? langs[0] : 'en')
    })
  }
})

// UI helpers
function getLanguageDisplayName(langCode) {
  const languageNames = {
    'en': 'English','es': 'Spanish','fr': 'French','de': 'German',
    'it': 'Italian','pt': 'Portuguese','pl': 'Polish','tr': 'Turkish',
    'ru': 'Russian','nl': 'Dutch','cs': 'Czech','ar': 'Arabic',
    'zh-cn': 'Chinese (Simplified)','hu': 'Hungarian','ko': 'Korean',
    'ja': 'Japanese','hi': 'Hindi'
  }
  return languageNames[langCode] || String(langCode).toUpperCase()
}

// Zero-shot upload
function triggerFileDialog() {
  fileInput.value?.click()
}

async function handleFileSelected(e) {
  const file = e.target.files?.[0]
  if (!file) return

  // Basic validation for WAV
  if (!file.name.toLowerCase().endsWith('.wav')) {
    errorMessage.value = 'Please upload a WAV file.'
    return
  }
  if (file.size > 20 * 1024 * 1024) {
    errorMessage.value = 'File is too large (max 20MB).'
    return
  }

  errorMessage.value = ''
  loading.value = true

  try {
    const form = new FormData()
    form.append('file', file, 'reference.wav')
    await axios.post(`${apiBase}/custom-voice/upload`, form, {
      headers: { 'Content-Type': 'multipart/form-data' }
    })
    // Reflect uploaded filename and mark zero-shot active
    zeroShotActive.value = true
    zeroShotName.value = file.name.replace(/\.[^/.]+$/, '')
    // After upload: refresh voices so zero-shot appears
    await loadVoices()
  } catch (err) {
    errorMessage.value = 'Upload failed: ' + (err.response?.data?.detail || err.message)
  } finally {
    loading.value = false
    if (fileInput.value) fileInput.value.value = ''
  }
}

async function clearZeroShot() {
  loading.value = true
  try {
    await axios.post(`${apiBase}/custom-voice/clear`)
    await loadVoices()
  } catch (e) {
    errorMessage.value = 'Failed to clear zero-shot: ' + (e.response?.data?.detail || e.message)
  } finally {
    loading.value = false
    zeroShotActive.value = false
    zeroShotName.value = ''
  }
}

// Synthesis
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
    if (selectedVoice.value.is_zero_shot) {
      // zero-shot endpoint (base checkpoint + uploaded reference)
      const payload = {
        text: text.value,
        language: selectedLanguage.value,
        speed: speed.value
      }
      const { data } = await axios.post(`${apiBase}/synthesize/zero-shot`, payload)
      audioUrl.value = data.audio_url
      filename.value = data.filename
      lastUsedVoice.value = 'Custom'
    } else {
      // fine-tuned voice endpoint
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
    }
  } catch (e) {
    console.error('Synthesis error:', e)
    console.error('Error response:', e.response?.data)
    errorMessage.value = 'Synthesis failed: ' + (e.response?.data?.detail || e.message)
  } finally {
    loading.value = false
  }
}

// Init
onMounted(async () => {
  await loadLanguages()
  await loadVoices()
  try {
    const { data } = await axios.get(`${apiBase}/gpu`)
    Object.assign(gpuInfo.value, data)
  } catch {
  }
  if (voices.value.length > 0 && !selectedVoice.value) {
    selectedVoice.value = voices.value[0]
    if (!selectedLanguage.value) {
      const langs = availableLanguagesForVoice.value
      selectedLanguage.value = selectedVoice.value.fixed_language || (langs.length ? langs[0] : 'en')
    }
  }
})
</script>
