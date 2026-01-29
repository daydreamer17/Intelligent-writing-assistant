<template>
  <div v-if="visible" class="progress-overlay">
    <div class="progress-card">
      <h3>{{ title }}</h3>
      <div class="progress-steps">
        <div
          v-for="(step, index) in steps"
          :key="index"
          class="progress-step"
          :class="{
            active: index === currentStep,
            completed: index < currentStep,
          }"
        >
          <div class="step-icon">
            <span v-if="index < currentStep">✓</span>
            <span v-else>{{ index + 1 }}</span>
          </div>
          <div class="step-label">{{ step }}</div>
        </div>
      </div>
      <div class="progress-bar">
        <div class="progress-fill" :style="{ width: progressPercent + '%' }"></div>
      </div>
      <div class="progress-text">{{ progressText }}</div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue';

interface Props {
  visible: boolean;
  title?: string;
  steps: string[];
  currentStep: number;
}

const props = withDefaults(defineProps<Props>(), {
  title: '处理中',
  visible: false,
  currentStep: 0,
});

const progressPercent = computed(() => {
  if (props.steps.length === 0) return 0;
  return Math.round((props.currentStep / props.steps.length) * 100);
});

const progressText = computed(() => {
  if (props.currentStep >= props.steps.length) {
    return '完成';
  }
  return `${props.currentStep} / ${props.steps.length} - ${props.steps[props.currentStep] || ''}`;
});
</script>

<style scoped>
.progress-overlay {
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: rgba(0, 0, 0, 0.7);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 9999;
}

.progress-card {
  background: white;
  border-radius: 8px;
  padding: 24px;
  min-width: 400px;
  max-width: 600px;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
}

.progress-card h3 {
  margin: 0 0 20px 0;
  font-size: 18px;
  text-align: center;
}

.progress-steps {
  display: flex;
  justify-content: space-between;
  margin-bottom: 24px;
  gap: 8px;
}

.progress-step {
  flex: 1;
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 8px;
}

.step-icon {
  width: 32px;
  height: 32px;
  border-radius: 50%;
  background: #e0e0e0;
  color: #666;
  display: flex;
  align-items: center;
  justify-content: center;
  font-weight: bold;
  font-size: 14px;
  transition: all 0.3s;
}

.progress-step.active .step-icon {
  background: #2196f3;
  color: white;
  animation: pulse 1.5s infinite;
}

.progress-step.completed .step-icon {
  background: #4caf50;
  color: white;
}

.step-label {
  font-size: 12px;
  text-align: center;
  color: #666;
}

.progress-step.active .step-label {
  color: #2196f3;
  font-weight: bold;
}

.progress-step.completed .step-label {
  color: #4caf50;
}

.progress-bar {
  width: 100%;
  height: 8px;
  background: #e0e0e0;
  border-radius: 4px;
  overflow: hidden;
  margin-bottom: 12px;
}

.progress-fill {
  height: 100%;
  background: linear-gradient(90deg, #2196f3, #4caf50);
  transition: width 0.3s ease;
}

.progress-text {
  text-align: center;
  font-size: 14px;
  color: #666;
}

@keyframes pulse {
  0%, 100% {
    transform: scale(1);
  }
  50% {
    transform: scale(1.1);
  }
}
</style>
