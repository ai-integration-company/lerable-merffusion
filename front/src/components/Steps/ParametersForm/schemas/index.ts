import { z } from 'zod';

export const ParametersFormSchema = z.object({
  param1: z.number(),
  param2: z.number(),
  param3: z.number(),
});
