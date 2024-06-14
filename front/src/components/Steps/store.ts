import { persistentAtom } from '@nanostores/persistent';
import { StepperFormData } from '@/types';

export const stepperFormStore = persistentAtom<StepperFormData | undefined>(
  'stepperForm',
  undefined,
  {
    encode: JSON.stringify,
    decode: JSON.parse,
  }
);
