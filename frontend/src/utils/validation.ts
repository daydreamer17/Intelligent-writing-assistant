/**
 * 表单验证工具函数
 */

export interface ValidationResult {
  valid: boolean;
  errors: string[];
}

/**
 * 验证必填字段
 */
export const validateRequired = (value: string | undefined, fieldName: string): string | null => {
  if (!value || value.trim() === '') {
    return `${fieldName}不能为空`;
  }
  return null;
};

/**
 * 验证最小长度
 */
export const validateMinLength = (value: string, minLength: number, fieldName: string): string | null => {
  if (value.length < minLength) {
    return `${fieldName}至少需要${minLength}个字符`;
  }
  return null;
};

/**
 * 验证最大长度
 */
export const validateMaxLength = (value: string, maxLength: number, fieldName: string): string | null => {
  if (value.length > maxLength) {
    return `${fieldName}不能超过${maxLength}个字符`;
  }
  return null;
};

/**
 * 验证URL格式
 */
export const validateUrl = (value: string, fieldName: string): string | null => {
  if (!value) return null; // 可选字段

  try {
    new URL(value);
    return null;
  } catch {
    return `${fieldName}格式不正确`;
  }
};

/**
 * 验证Pipeline请求
 */
export const validatePipelineRequest = (form: {
  topic?: string;
  audience?: string;
  style?: string;
  target_length?: string;
}): ValidationResult => {
  const errors: string[] = [];

  // 验证必填字段
  const topicError = validateRequired(form.topic, '主题');
  if (topicError) errors.push(topicError);

  const audienceError = validateRequired(form.audience, '目标读者');
  if (audienceError) errors.push(audienceError);

  // 验证长度
  if (form.topic && form.topic.length < 3) {
    errors.push('主题至少需要3个字符');
  }

  if (form.topic && form.topic.length > 200) {
    errors.push('主题不能超过200个字符');
  }

  return {
    valid: errors.length === 0,
    errors,
  };
};

/**
 * 验证草稿请求
 */
export const validateDraftRequest = (form: {
  topic?: string;
  outline?: string;
}): ValidationResult => {
  const errors: string[] = [];

  const topicError = validateRequired(form.topic, '主题');
  if (topicError) errors.push(topicError);

  const outlineError = validateRequired(form.outline, '大纲');
  if (outlineError) errors.push(outlineError);

  return {
    valid: errors.length === 0,
    errors,
  };
};

/**
 * 验证审校请求
 */
export const validateReviewRequest = (form: {
  draft?: string;
}): ValidationResult => {
  const errors: string[] = [];

  const draftError = validateRequired(form.draft, '草稿');
  if (draftError) errors.push(draftError);

  return {
    valid: errors.length === 0,
    errors,
  };
};

/**
 * 验证改写请求
 */
export const validateRewriteRequest = (form: {
  draft?: string;
  guidance?: string;
}): ValidationResult => {
  const errors: string[] = [];

  const draftError = validateRequired(form.draft, '草稿');
  if (draftError) errors.push(draftError);

  const guidanceError = validateRequired(form.guidance, '修改指导');
  if (guidanceError) errors.push(guidanceError);

  return {
    valid: errors.length === 0,
    errors,
  };
};

/**
 * 验证文件大小
 */
export const validateFileSize = (file: File, maxSizeMB: number = 10): string | null => {
  const maxSizeBytes = maxSizeMB * 1024 * 1024;
  if (file.size > maxSizeBytes) {
    return `文件"${file.name}"过大，最大支持${maxSizeMB}MB`;
  }
  return null;
};

/**
 * 验证文件类型
 */
export const validateFileType = (file: File, allowedTypes: string[]): string | null => {
  const extension = file.name.split('.').pop()?.toLowerCase();
  if (!extension || !allowedTypes.includes(extension)) {
    return `文件"${file.name}"类型不支持，仅支持：${allowedTypes.join(', ')}`;
  }
  return null;
};

/**
 * 批量验证文件
 */
export const validateFiles = (files: File[]): ValidationResult => {
  const errors: string[] = [];
  const allowedTypes = ['txt', 'pdf', 'docx'];
  if (!files.length) {
    errors.push('请先选择文件');
  }

  for (const file of files) {
    const sizeError = validateFileSize(file);
    if (sizeError) errors.push(sizeError);

    const typeError = validateFileType(file, allowedTypes);
    if (typeError) errors.push(typeError);
  }

  return {
    valid: errors.length === 0,
    errors,
  };
};
