// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package intercepts_test

import (
	"encoding/json"

	"github.com/NVIDIA/NeMo-Relay/go/nemo_relay"
)

func requestForTest(value any) nemo_relay.LLMRequest {
	payload, err := json.Marshal(value)
	if err != nil {
		panic(err)
	}
	var request nemo_relay.LLMRequest
	if err := json.Unmarshal(payload, &request); err != nil {
		panic(err)
	}
	return request
}
