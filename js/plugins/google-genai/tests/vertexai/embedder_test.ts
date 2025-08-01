/**
 * Copyright 2025 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

import * as assert from 'assert';
import { Document, Genkit, GENKIT_CLIENT_HEADER } from 'genkit';
import { GoogleAuth } from 'google-auth-library';
import { afterEach, beforeEach, describe, it } from 'node:test';
import * as sinon from 'sinon';
import { getVertexAIUrl } from '../../src/vertexai/client';
import { defineEmbedder, EmbeddingConfig } from '../../src/vertexai/embedder';
import {
  ClientOptions,
  EmbedContentResponse,
  EmbeddingInstance,
} from '../../src/vertexai/types';

describe('defineEmbedder', () => {
  let mockGenkit: sinon.SinonStubbedInstance<Genkit>;
  let fetchStub: sinon.SinonStub;
  let authMock: sinon.SinonStubbedInstance<GoogleAuth>;

  const regionalClientOptions: ClientOptions = {
    kind: 'regional',
    projectId: 'test-project',
    location: 'us-central1',
    authClient: {} as GoogleAuth,
  };

  const globalClientOptions: ClientOptions = {
    kind: 'global',
    projectId: 'test-project',
    location: 'global',
    authClient: {} as GoogleAuth,
    apiKey: 'test-global-api-key',
  };

  let embedderFunc: (
    input: Document[],
    options?: EmbeddingConfig
  ) => Promise<any>;

  beforeEach(() => {
    mockGenkit = sinon.createStubInstance(Genkit);
    fetchStub = sinon.stub(global, 'fetch');

    authMock = sinon.createStubInstance(GoogleAuth);
    authMock.getAccessToken.resolves('test-token');
    regionalClientOptions.authClient = authMock as unknown as GoogleAuth;
    globalClientOptions.authClient = authMock as unknown as GoogleAuth;

    mockGenkit.defineEmbedder.callsFake((config, func) => {
      embedderFunc = func;
      return {
        name: config.name,
      } as any;
    });
  });

  afterEach(() => {
    sinon.restore();
  });

  function mockFetchResponse(body: any, status = 200) {
    const response = new Response(JSON.stringify(body), {
      status: status,
      statusText: 'OK',
      headers: { 'Content-Type': 'application/json' },
    });
    fetchStub.resolves(response);
  }

  function getExpectedHeaders(
    clientOptions: ClientOptions
  ): Record<string, string | undefined> {
    const headers: Record<string, string | undefined> = {
      'Content-Type': 'application/json',
      'X-Goog-Api-Client': GENKIT_CLIENT_HEADER,
      'User-Agent': GENKIT_CLIENT_HEADER,
      Authorization: 'Bearer test-token',
      'x-goog-user-project':
        clientOptions.kind != 'express' ? clientOptions.projectId : '',
    };
    if (clientOptions.apiKey) {
      headers['x-goog-api-key'] = clientOptions.apiKey;
    }
    return headers;
  }

  function runTestsForClientOptions(clientOptions: ClientOptions) {
    describe(`with ${clientOptions.kind} client options`, () => {
      it('defines an embedder with the correct name and info for known model', () => {
        defineEmbedder(mockGenkit, 'text-embedding-005', clientOptions);
        sinon.assert.calledOnce(mockGenkit.defineEmbedder);
        const args = mockGenkit.defineEmbedder.lastCall.args[0];
        assert.strictEqual(args.name, 'vertexai/text-embedding-005');
        assert.strictEqual(args.info?.dimensions, 768);
      });

      it('defines an embedder with a custom name', () => {
        defineEmbedder(mockGenkit, 'custom-model', clientOptions);
        sinon.assert.calledOnce(mockGenkit.defineEmbedder);
        const args = mockGenkit.defineEmbedder.lastCall.args[0];
        assert.strictEqual(args.name, 'vertexai/custom-model');
      });

      describe('Embedder Functionality', () => {
        const testDoc1: Document = new Document({
          content: [{ text: 'Hello' }],
        });
        const testDoc2: Document = new Document({
          content: [{ text: 'World' }],
        });

        it('calls embedContent with text-only documents', async () => {
          defineEmbedder(mockGenkit, 'text-embedding-005', clientOptions);

          const mockResponse: EmbedContentResponse = {
            predictions: [
              {
                embeddings: {
                  values: [0.1, 0.2],
                  statistics: { token_count: 1, truncated: false },
                },
              },
              {
                embeddings: {
                  values: [0.3, 0.4],
                  statistics: { token_count: 1, truncated: false },
                },
              },
            ],
          };
          mockFetchResponse(mockResponse);

          const result = await embedderFunc([testDoc1, testDoc2]);

          sinon.assert.calledOnce(fetchStub);
          const fetchArgs = fetchStub.lastCall.args;
          const expectedUrl = getVertexAIUrl({
            includeProjectAndLocation: true,
            resourcePath: 'publishers/google/models/text-embedding-005',
            resourceMethod: 'predict',
            clientOptions,
          });
          assert.strictEqual(fetchArgs[0], expectedUrl);

          const expectedRequest = {
            instances: [{ content: 'Hello' }, { content: 'World' }],
            parameters: {}, // Undefined properties are omitted
          };
          assert.deepStrictEqual(
            JSON.parse(fetchArgs[1].body),
            expectedRequest
          );
          assert.deepStrictEqual(
            fetchArgs[1].headers,
            getExpectedHeaders(clientOptions)
          );

          assert.deepStrictEqual(result, {
            embeddings: [{ embedding: [0.1, 0.2] }, { embedding: [0.3, 0.4] }],
          });
        });

        it('calls embedContent with taskType and title options', async () => {
          defineEmbedder(mockGenkit, 'text-embedding-005', clientOptions);
          mockFetchResponse({ predictions: [] });

          const config: EmbeddingConfig = {
            taskType: 'RETRIEVAL_DOCUMENT',
            title: 'Doc Title',
          };
          await embedderFunc([testDoc1], config);

          sinon.assert.calledOnce(fetchStub);
          const fetchOptions = fetchStub.lastCall.args[1];
          const body = JSON.parse(fetchOptions.body);
          assert.strictEqual(body.instances[0].task_type, 'RETRIEVAL_DOCUMENT');
          assert.strictEqual(body.instances[0].title, 'Doc Title');
        });

        it('handles multimodal embeddings for images (base64)', async () => {
          defineEmbedder(mockGenkit, 'multimodalembedding@001', clientOptions);
          const docWithImage: Document = new Document({
            content: [
              { text: 'A picture' },
              { media: { url: 'base64data', contentType: 'image/png' } },
            ],
          });

          const mockResponse: EmbedContentResponse = {
            predictions: [{ textEmbedding: [0.1], imageEmbedding: [0.2] }],
          };
          mockFetchResponse(mockResponse);

          const result = await embedderFunc([docWithImage]);

          const expectedInstance: EmbeddingInstance = {
            text: 'A picture',
            image: { bytesBase64Encoded: 'base64data', mimeType: 'image/png' },
          };
          const fetchBody = JSON.parse(fetchStub.lastCall.args[1].body);
          assert.deepStrictEqual(fetchBody.instances[0], expectedInstance);
          assert.deepStrictEqual(result.embeddings.length, 2);
        });

        it('handles multimodal embeddings for images (gcs)', async () => {
          defineEmbedder(mockGenkit, 'multimodalembedding@001', clientOptions);
          const docWithImage: Document = new Document({
            content: [
              {
                media: {
                  url: 'gs://bucket/image.jpg',
                  contentType: 'image/jpeg',
                },
              },
            ],
          });
          mockFetchResponse({ predictions: [] });
          await embedderFunc([docWithImage]);

          const expectedInstance: EmbeddingInstance = {
            image: { gcsUri: 'gs://bucket/image.jpg', mimeType: 'image/jpeg' },
          };
          const fetchBody = JSON.parse(fetchStub.lastCall.args[1].body);
          assert.deepStrictEqual(fetchBody.instances[0], expectedInstance);
        });

        it('passes outputDimensionality to the API call', async () => {
          defineEmbedder(mockGenkit, 'text-embedding-005', clientOptions);
          mockFetchResponse({ predictions: [] });

          const config: EmbeddingConfig = { outputDimensionality: 256 };
          await embedderFunc([testDoc1], config);

          sinon.assert.calledOnce(fetchStub);
          const fetchOptions = fetchStub.lastCall.args[1];
          const body = JSON.parse(fetchOptions.body);
          assert.strictEqual(body.parameters.outputDimensionality, 256);
        });
      });
    });
  }

  runTestsForClientOptions(regionalClientOptions);
  runTestsForClientOptions(globalClientOptions);
  // Express client options does not support embedders. We have
  // tests elsewhere to test this.

  // Tests specific to regional (or not applicable to express)
  describe('with regional client options only', () => {
    const clientOptions = regionalClientOptions;
    it('handles multimodal embeddings for video', async () => {
      defineEmbedder(mockGenkit, 'multimodalembedding@001', clientOptions);
      const docWithVideo: Document = new Document({
        content: [
          { text: 'A video' },
          { media: { url: 'base64video', contentType: 'video/mp4' } },
        ],
        metadata: {
          videoSegmentConfig: {
            startOffsetSec: 0,
            endOffsetSec: 10,
            intervalSec: 5,
          },
        },
      });

      const mockResponse: EmbedContentResponse = {
        predictions: [
          {
            textEmbedding: [0.1],
            videoEmbeddings: [
              { startOffsetSec: 0, endOffsetSec: 5, embedding: [0.8, 0.9] },
              { startOffsetSec: 5, endOffsetSec: 10, embedding: [0.6, 0.7] },
            ],
          },
        ],
      };
      mockFetchResponse(mockResponse);

      const result = await embedderFunc([docWithVideo]);

      const expectedInstance: EmbeddingInstance = {
        text: 'A video',
        video: {
          bytesBase64Encoded: 'base64video',
          videoSegmentConfig: {
            startOffsetSec: 0,
            endOffsetSec: 10,
            intervalSec: 5,
          },
        },
      };
      const fetchBody = JSON.parse(fetchStub.lastCall.args[1].body);
      assert.deepStrictEqual(fetchBody.instances[0], expectedInstance);

      assert.deepStrictEqual(result, {
        embeddings: [
          { embedding: [0.1], metadata: { embedType: 'textEmbedding' } },
          {
            embedding: [0.8, 0.9],
            metadata: {
              startOffsetSec: 0,
              endOffsetSec: 5,
              embedType: 'videoEmbedding',
            },
          },
          {
            embedding: [0.6, 0.7],
            metadata: {
              startOffsetSec: 5,
              endOffsetSec: 10,
              embedType: 'videoEmbedding',
            },
          },
        ],
      });
    });

    it('throws on unsupported media type', async () => {
      defineEmbedder(mockGenkit, 'multimodalembedding@001', clientOptions);
      const docWithInvalidMedia: Document = new Document({
        content: [{ media: { url: 'a', contentType: 'application/pdf' } }],
      });
      await assert.rejects(
        embedderFunc([docWithInvalidMedia]),
        /Unsupported contentType: 'application\/pdf/
      );
      sinon.assert.notCalled(fetchStub);
    });
  });
});
