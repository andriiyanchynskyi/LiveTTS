<template>
  <main style="max-width: 880px; margin: 24px auto; font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif;">
    <h1>Coqui VITS — Local TTS</h1>

    <section style="margin: 16px 0; padding: 12px; border: 1px solid #ddd; border-radius: 12px;">
      <label style="display:block; font-weight:600; margin-bottom:8px;">Text</label>
      <textarea v-model="text" rows="5" style="width:100%; padding:10px;"></textarea>

      <div style="display:flex; gap:16px; margin-top:12px; align-items:center; flex-wrap: wrap;">
        <div>
          <label style="display:block; font-weight:600; margin-bottom:6px;">Voice</label>
          <select v-model="speaker">
            <option v-for="s in speakers" :key="s" :value="s">{{ s }}</option>
          </select>
        </div>

        <div>
          <label style="display:block; font-weight:600; margin-bottom:6px;">Language</label>
          <select v-model="language">
            <option value="">(auto/none)</option>
            <option v-for="l in languages" :key="l" :value="l">{{ l }}</option>
          </select>
        </div>

        <button :disabled="loading || !text.trim()" @click="synthesize"
                style="padding:8px 14px; border-radius:10px; border:1px solid #333; cursor:pointer;">
          {{ loading ? "Generating..." : "Synthesize" }}
        </button>
      </div>
    </section>

    <section v-if="audioUrl" style="margin-top:16px; padding: 12px; border: 1px solid #ddd; border-radius: 12px;">
      <h3>Result</h3>
      <audio :src="apiBase + audioUrl" controls style="width:100%;"></audio>
      <div style="margin-top:8px;">
        <a :href="apiBase + audioUrl" :download="filename">Download WAV</a>
      </div>
    </section>

    <section style="margin-top:24px; color:#666;">
      <details>
        <summary>GPU Diagnostics</summary>
        <pre>{{ gpuInfo }}</pre>
      </details>
    </section>
  </main>
</template>

<script setup>
import axios from 'axios'
import { ref } from 'vue'

// In Docker Compose we pass VITE_API_BASE=http://localhost:8000
const apiBase = import.meta.env.VITE_API_BASE || 'http://localhost:8000'

const text = ref('Hello! This is local speech synthesis based on Coqui TTS and VITS model.')
const speakers = ref([])
const languages = ref([])
const speaker = ref('')
const language = ref('')
const audioUrl = ref('')
const filename = ref('')
const loading = ref(false)
const gpuInfo = ref({})

async function loadVoices() {
  const { data } = await axios.get(`${apiBase}/voices`)
  speakers.value.splice(0, speakers.value.length, ...(data.speakers || ['default']))
  languages.value.splice(0, languages.value.length, ...(data.languages || []))
  if (!speaker.value && speakers.value.length) {
    speaker.value = speakers.value[0]
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

async function synthesize() {
  loading.value = true
  audioUrl.value = ''
  filename.value = ''
  try {
    const payload = { text: text.value, speaker: speaker.value || null, language: language.value || null }
    const { data } = await axios.post(`${apiBase}/synthesize`, payload)
    audioUrl.value = data.url
    filename.value = data.filename
  } catch (e) {
    alert('Synthesis error: ' + e)
  } finally {
    loading.value = false
  }
}

loadVoices()
diagGpu()
</script>

<style>
/* Small shingle for reactivity macros $ref (Vite + Vue Macros not used) */
</style>
