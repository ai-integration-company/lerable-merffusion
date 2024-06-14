import { useNavigate } from 'react-router-dom';
import { notifications } from '@mantine/notifications';
import { LoadingOverlay, rem } from '@mantine/core';
import { useState } from 'react';
import { Actions, ProductForm, ProductFormData, Stepper } from '@/components';
import { getBackground } from '@/api';
import { resultStore } from '@/store';

export const HomePage = () => {
  const navigate = useNavigate();
  const [loading, setLoading] = useState(false);
  const onSubmit = async (data: ProductFormData) => {
    setLoading(true);
    try {
      const response = await getBackground(data.productPicture.fileData);
      resultStore.set(response);
      navigate('/results');
    } catch (error) {
      console.log(error);
      notifications.show({
        color: 'red',
        title: 'Произошла ошибка при генерации фона',
        message: 'Попробуйте пожалуйста позже',
      });
    } finally {
      setLoading(false);
    }
  };
  return (
    <Stepper>
      <ProductForm onSubmit={onSubmit} actions={<Actions backButtonDisabled />} />
      <LoadingOverlay
        visible={loading}
        zIndex={1000}
        overlayProps={{ blur: 2 }}
        style={{ borderRadius: rem(16) }}
      />
    </Stepper>
  );
};
