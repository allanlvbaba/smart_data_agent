/*
 * Copyright 2024-2026 the original author or authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package com.alibaba.cloud.ai.dataagent.service.llm.impls;

import com.alibaba.cloud.ai.dataagent.service.aimodelconfig.AiModelRegistry;
import com.alibaba.cloud.ai.dataagent.service.llm.LlmService;
import lombok.AllArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.ai.chat.client.advisor.StructuredOutputValidationAdvisor;
import org.springframework.ai.chat.model.ChatResponse;
import reactor.core.publisher.Flux;
import reactor.core.publisher.Mono;
import reactor.core.scheduler.Schedulers;

@Slf4j
@AllArgsConstructor
public class StreamLlmService implements LlmService {

	private final AiModelRegistry registry;

	@Override
	public Flux<ChatResponse> call(String system, String user) {
		traceLLMCall("call(system,user)", system, user);
		return registry.getChatClient().prompt().system(system).user(user).stream().chatResponse();
	}

	@Override
	public Flux<ChatResponse> call(String system, String user, Class<?> outputType) {
		traceLLMCall("call(system,user,type)", system, user);
		StructuredOutputValidationAdvisor advisor = StructuredOutputValidationAdvisor.builder()
			.outputType(outputType)
			.maxRepeatAttempts(2)
			.build();
		return Mono
			.fromCallable(() -> registry.getChatClient()
				.prompt()
				.system(system)
				.user(user)
				.advisors(advisor)
				.call()
				.chatResponse())
			.subscribeOn(Schedulers.boundedElastic())
			.flux();
	}

	@Override
	public Flux<ChatResponse> callSystem(String system) {
		traceLLMCall("callSystem(only-system)", system, null);
		return registry.getChatClient().prompt().system(system).stream().chatResponse();
	}

	@Override
	public Flux<ChatResponse> callUser(String user) {
		traceLLMCall("callUser(only-user)", null, user);
		return registry.getChatClient().prompt().user(user).stream().chatResponse();
	}

	@Override
	public Flux<ChatResponse> callUser(String user, Class<?> outputType) {
		traceLLMCall("callUser(user,type)", null, user);
		StructuredOutputValidationAdvisor advisor = StructuredOutputValidationAdvisor.builder()
			.outputType(outputType)
			.maxRepeatAttempts(2)
			.build();
		return Mono
			.fromCallable(() -> registry.getChatClient().prompt().user(user).advisors(advisor).call().chatResponse())
			.subscribeOn(Schedulers.boundedElastic())
			.flux();
	}

	/**
	 * 临时诊断：定位智谱 1214（messages 参数非法）的调用来源。重点关注 blank 消息与仅有
	 * system 无 user 的调用（OpenAI 允许、智谱拒绝）。问题闭环后可整体移除。
	 */
	private void traceLLMCall(String method, String system, String user) {
		int sysLen = system == null ? -1 : system.length();
		int usrLen = user == null ? -1 : user.length();
		if (sysLen == 0 || usrLen == 0) {
			log.warn("[LLM-TRACE] {} called with BLANK message (systemLen={}, userLen={})", method, sysLen, usrLen,
					new Exception("call-site-stack"));
		}
		else if ("callSystem(only-system)".equals(method)) {
			log.warn("[LLM-TRACE] {} no user message (systemLen={}) —— 智谱可能拒绝该结构", method, sysLen,
					new Exception("call-site-stack"));
		}
		else {
			log.info("[LLM-TRACE] {} systemLen={}, userLen={}", method, sysLen, usrLen);
		}
	}

}
