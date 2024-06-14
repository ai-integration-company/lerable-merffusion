import { persistentAtom } from '@nanostores/persistent';
import { ApiResponse, StepperFormData } from '@/types';

export const stepperFormStore = persistentAtom<
  StepperFormData | undefined | { productPicture: undefined }
>(
  'stepperForm',
  { productPicture: undefined },
  {
    encode: JSON.stringify,
    decode: JSON.parse,
  }
);

export const resultStore = persistentAtom<ApiResponse | undefined>('result', undefined, {
  encode: JSON.stringify,
  decode: JSON.parse,
});
