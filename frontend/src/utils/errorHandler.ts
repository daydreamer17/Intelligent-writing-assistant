import axios from 'axios';

/**
 * 统一的API错误处理函数
 * @param err 错误对象
 * @returns 用户友好的错误消息
 */
export const handleApiError = (err: unknown): string => {
  // 处理Axios错误
  if (axios.isAxiosError(err)) {
    const status = err.response?.status;
    const detail = err.response?.data?.detail;

    // 根据HTTP状态码返回不同的错误消息
    switch (status) {
      case 400:
        return `请求错误：${detail || '参数不正确'}`;
      case 401:
        return '未授权，请检查API密钥';
      case 403:
        return '禁止访问';
      case 404:
        return '请求的资源不存在';
      case 413:
        return '文件过大（最大10MB）';
      case 422:
        return `参数验证失败：${detail || '请检查输入'}`;
      case 429:
        return '触发TPM限流，请稍后重试或降低字数/并发';
      case 500:
        return `服务器错误：${detail || '请稍后重试'}`;
      case 502:
        return '网关错误，后端服务可能未启动';
      case 503:
        return '服务暂时不可用';
      case 504:
        return '请求超时';
      default:
        return detail || err.message || '未知错误';
    }
  }

  // 处理网络错误
  if (err instanceof Error) {
    if (err.message.includes('Network Error')) {
      return '网络错误，请检查后端服务是否启动（默认端口8000）';
    }
    if (err.message.includes('timeout')) {
      return '请求超时，请检查网络连接';
    }
    return err.message;
  }

  // 处理其他类型的错误
  return '操作失败，请重试';
};

/**
 * 错误类型枚举
 */
export enum ErrorType {
  NETWORK = 'network',
  VALIDATION = 'validation',
  SERVER = 'server',
  UNKNOWN = 'unknown',
}

/**
 * 获取错误类型
 * @param err 错误对象
 * @returns 错误类型
 */
export const getErrorType = (err: unknown): ErrorType => {
  if (axios.isAxiosError(err)) {
    const status = err.response?.status;

    if (!status) {
      return ErrorType.NETWORK;
    }

    if (status >= 400 && status < 500) {
      return ErrorType.VALIDATION;
    }

    if (status >= 500) {
      return ErrorType.SERVER;
    }
  }

  return ErrorType.UNKNOWN;
};
