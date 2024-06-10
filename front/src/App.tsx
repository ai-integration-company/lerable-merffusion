import '@mantine/core/styles.css';
import { Flex, MantineProvider } from '@mantine/core';
import { Notifications } from '@mantine/notifications';
import { Router } from './Router';
import { theme } from './theme';
import '@mantine/notifications/styles.css';

export default function App() {
  return (
    <MantineProvider theme={theme}>
      <Notifications position="top-right" />
      <Flex
        style={{
          width: '100vw',
          minHeight: 'fit-content',
          justifyContent: 'center',
          alignItems: 'center',
        }}
      >
        <Flex
          direction="column"
          align="center"
          gap="lg"
          w="100%"
          style={{ minHeight: '100vh' }}
          justify="center"
          py="md"
        >
          <Router />
        </Flex>
      </Flex>
    </MantineProvider>
  );
}
