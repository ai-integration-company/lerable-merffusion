import { z } from 'zod';

export const ProductFormSchema = z
  .object({
    productPicture: z.object({
      fileData: z.string(),
      fileName: z.string(),
      mimeType: z.string(),
    }),
  })
  .refine(
    (data) => {
      if (!data.productPicture.fileData) {
        return false;
      }
      return true;
    },
    {
      message: 'Необходимо загрузить фотографию товара',
      path: ['productPicture'],
    }
  );
