import axios from 'axios'
import { ref, onMounted, computed, watch, nextTick } from 'vue'

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
// Centered slider value (0–100), 50 corresponds to 1.0x
const sliderValue = ref(50)
// Map sliderValue -> speed: 0.5x–1.0x on left half, 1.0x–2.0x on right
const speed = computed(() => {
  const pos = sliderValue.value / 100
  let raw
  if (pos <= 0.5) {
    raw = 0.5 + pos 
  } else {
    raw = 1.0 + (pos - 0.5) * 2 // 
  }
  // Round to one decimal (
  return Math.round(raw * 10) / 10
})

// Zero-shot helpers
const fileInput = ref(null)
const showTooltip = ref(false)
const zeroShotActive = ref(false)
const zeroShotName = ref('')

// Enhancement states (global)
const enhancementStates = ref({})

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
        speed: speed.value,
        enhancements: { ...enhancementStates.value }  // shallow clone
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
        speed: speed.value,
        enhancements: { ...enhancementStates.value }  // shallow clone
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

// Init function
async function initializeApp() {
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
}

// Composable to fetch and manage audio enhancements
 function useEnhancements(apiBase) {
  const enhancements = ref({})  // {key: description}
  const activeTooltip = ref(null)  // key of current hovered toggle

  // Fetch enhancements on mount
  onMounted(async () => {
    try {
      const { data } = await axios.get(`${apiBase}/enhancements`)
      enhancements.value = data
      // Initialize states to false
      Object.keys(data).forEach(key => {
        enhancementStates.value[key] = false
      })
    } catch (e) {
      console.error('Failed to load enhancements:', e)
    }
  })

  // Tooltip handlers
  function setTooltip(key, desc) {
    activeTooltip.value = key
  }
  function clearTooltip() {
    activeTooltip.value = null
  }

  return { enhancements, enhancementStates, activeTooltip, setTooltip, clearTooltip }
}

// Export all reactive state and functions
export {
  // Constants
  apiBase,
  
  // Reactive state
  text,
  voices,
  selectedLanguage,
  selectedVoice,
  audioUrl,
  filename,
  lastUsedVoice,
  loading,
  refreshing,
  errorMessage,
  systemStatus,
  gpuInfo,
  availableLanguages,
  sliderValue,
  speed,
  
  // Zero-shot helpers
  fileInput,
  showTooltip,
  zeroShotActive,
  zeroShotName,
  
  // Enhancement states
  enhancementStates,
  
  // Computed
  availableLanguagesForVoice,
  isReadyForSynthesis,
  
  // Functions
  loadLanguages,
  loadVoices,
  refreshVoices,
  diagGpu,
  getLanguageDisplayName,
  triggerFileDialog,
  handleFileSelected,
  clearZeroShot,
  synthesize,
  initializeApp,
  
  // Composables
  useEnhancements
}
