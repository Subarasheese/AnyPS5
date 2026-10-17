#include "SpirvBackend/SpirvBda.hpp"
#include "RuntimeAbi.hpp"
#include "SpirvBackend/SpirvMemory/SpirvTypes.hpp"
#include "SpirvBackend/SpirvMemory/SpirvConstants.hpp"
#include <bit>
#include <utility>
#include <iterator>
#include <initializer_list>

namespace ShaderRecompiler {

namespace {

void ReturnBdaZeroIf(SpirvEmitterState& state, std::uint32_t condition) {
    const auto failed = state.module.AllocateId();
    const auto next = state.module.AllocateId();
    state.module.AddFunction(spv::OpSelectionMerge, next, spv::SelectionControlMaskNone);
    state.module.AddFunction(spv::OpBranchConditional, condition, failed, next);
    EmitLabel(state, failed);
    state.module.AddFunction(spv::OpReturnValue, BdaConstant(state, 0u));
    EmitLabel(state, next);
}

// (u64 address, u32 bytes, u32 instruction) -> u64 device address of the range holding all the bytes,
// GPU-resolved programs (the native renderer): RuntimeAbi::NativePageTable, a direct two-level
// table instead of the binary search over ranges.
std::uint32_t DefineNativePageLookup(SpirvEmitterState& state, const char* name, bool recordFaults, std::uint32_t permission) {
    namespace Table = RuntimeAbi::NativePageTable;
    const auto u32 = TypeU32(state);
    const auto u64 = TypeScalarU64(state);
    const auto boolean = TypeBool(state);
    const auto constant = [&](std::uint32_t value) { return ConstantU32(state, value); };
    const auto binary = [&](std::uint32_t op, std::uint32_t type, std::uint32_t left, std::uint32_t right) { return Binary(state, op, type, left, right); };
    const auto function = state.module.AllocateId();
    state.module.AddName(function, name);
    state.module.AddFunction(spv::OpFunction, u64, function, spv::FunctionControlDontInlineMask, state.module.Type(spv::OpTypeFunction, u64, u64, u32, u32));
    const auto address = state.module.AllocateId();
    const auto bytes = state.module.AllocateId();
    const auto instruction = state.module.AllocateId();
    state.module.AddFunction(spv::OpFunctionParameter, u64, address);
    state.module.AddFunction(spv::OpFunctionParameter, u32, bytes);
    state.module.AddFunction(spv::OpFunctionParameter, u32, instruction);
    EmitLabel(state, state.module.AllocateId());
    const auto fail = [&](std::uint32_t condition, BdaAbi::FaultReason reason) {
        if (recordFaults) ReturnBdaFailureIf(state, condition, address, bytes, instruction, reason);
        else ReturnBdaZeroIf(state, condition);
    };
    const auto low = Unary(state, spv::OpUConvert, u32, address);
    fail(binary(spv::OpINotEqual, boolean, BdaLoadWord(state, constant(0)), constant(Table::Magic)), BdaAbi::FaultReason::InvalidTable);
    const auto level1Count = BdaLoadWord(state, constant(1));
    const auto region = Unary(state, spv::OpUConvert, u32, binary(spv::OpShiftRightLogical, u64, address, BdaConstant(state, Table::BlockShift)));
    const auto above = binary(spv::OpUGreaterThanEqual, boolean, binary(spv::OpShiftRightLogical, u64, address, BdaConstant(state, Table::BlockShift)), Unary(state, spv::OpUConvert, u64, level1Count));
    fail(above, BdaAbi::FaultReason::Unmapped);
    const auto block = BdaLoadWord(state, binary(spv::OpIAdd, u32, constant(Table::HeaderWords), region));
    fail(binary(spv::OpIEqual, boolean, block, constant(0)), BdaAbi::FaultReason::Unmapped);
    const auto page = binary(spv::OpBitwiseAnd, u32, binary(spv::OpShiftRightLogical, u32, low, constant(Table::PageShift)), constant(Table::PagesPerBlock - 1u));
    const auto level2 = binary(spv::OpIAdd, u32, binary(spv::OpBitwiseAnd, u32, binary(spv::OpIAdd, u32, constant(Table::HeaderWords + 3u), level1Count), constant(~3u)), binary(spv::OpIMul, u32, binary(spv::OpISub, u32, block, constant(1u)), constant(Table::PagesPerBlock * 4u)));
    const auto entry = binary(spv::OpIAdd, u32, level2, binary(spv::OpIMul, u32, page, constant(4u)));
    const auto device = BdaLoadAddress(state, entry);
    fail(binary(spv::OpIEqual, boolean, device, BdaConstant(state, 0)), BdaAbi::FaultReason::Unmapped);
    const auto validBegin = BdaLoadWord(state, binary(spv::OpIAdd, u32, entry, constant(2u)));
    const auto validWord = BdaLoadWord(state, binary(spv::OpIAdd, u32, entry, constant(3u)));
    const auto offset = binary(spv::OpBitwiseAnd, u32, low, constant((1u << Table::PageShift) - 1u));
    fail(binary(spv::OpULessThan, boolean, offset, validBegin), BdaAbi::FaultReason::Unmapped);
    const auto validBytes = binary(spv::OpBitwiseAnd, u32, validWord, constant(Table::BytesMask));
    const auto end = binary(spv::OpIAdd, u64, Unary(state, spv::OpUConvert, u64, offset), Unary(state, spv::OpUConvert, u64, bytes));
    fail(binary(spv::OpUGreaterThan, boolean, end, Unary(state, spv::OpUConvert, u64, validBytes)), BdaAbi::FaultReason::Unmapped);
    const auto permissions = binary(spv::OpShiftRightLogical, u32, validWord, constant(Table::PermissionShift));
    fail(binary(spv::OpINotEqual, boolean, binary(spv::OpBitwiseAnd, u32, permissions, constant(permission)), constant(permission)), BdaAbi::FaultReason::Permission);
    state.module.AddFunction(spv::OpReturnValue, binary(spv::OpIAdd, u64, device, Unary(state, spv::OpUConvert, u64, offset)));
    state.module.AddFunction(spv::OpFunctionEnd);
    return function;
}

std::uint32_t DefineBdaLookup(SpirvEmitterState& state, const char* name, bool recordFaults, std::uint32_t permission = BdaAbi::Read) {
    if (state.program.Metadata().gpuResolvedResources) return DefineNativePageLookup(state, name, recordFaults, permission);
    const auto u32 = TypeU32(state);
    const auto u64 = TypeScalarU64(state);
    const auto boolean = TypeBool(state);
    const auto constant = [&](std::uint32_t value) { return ConstantU32(state, value); };
    const auto binary = [&](std::uint32_t op, std::uint32_t type, std::uint32_t left, std::uint32_t right) { return Binary(state, op, type, left, right); };
    const auto function = state.module.AllocateId();
    state.module.AddName(function, name);
    state.module.AddFunction(spv::OpFunction, u64, function, spv::FunctionControlDontInlineMask, state.module.Type(spv::OpTypeFunction, u64, u64, u32, u32));
    const auto address = state.module.AllocateId();
    const auto bytes = state.module.AllocateId();
    const auto instruction = state.module.AllocateId();
    state.module.AddFunction(spv::OpFunctionParameter, u64, address);
    state.module.AddFunction(spv::OpFunctionParameter, u32, bytes);
    state.module.AddFunction(spv::OpFunctionParameter, u32, instruction);
    EmitLabel(state, state.module.AllocateId());
    const auto fail = [&](std::uint32_t condition, BdaAbi::FaultReason reason) {
        if (recordFaults) {
            ReturnBdaFailureIf(state, condition, address, bytes, instruction, reason);
        } else {
            ReturnBdaZeroIf(state, condition);
        }
    };
    const auto failFirst = [&](std::initializer_list<std::pair<std::uint32_t, BdaAbi::FaultReason>> checks) {
        auto reason = constant(0u);
        for (auto check = std::rbegin(checks); check != std::rend(checks); ++check) {
            reason = Select(state, u32, check->first, constant(static_cast<std::uint32_t>(check->second)), reason);
        }
        const auto failed = binary(spv::OpINotEqual, boolean, reason, constant(0u));
        if (!recordFaults) {
            ReturnBdaZeroIf(state, failed);
            return;
        }
        const auto failedLabel = state.module.AllocateId();
        const auto next = state.module.AllocateId();
        state.module.AddFunction(spv::OpSelectionMerge, next, spv::SelectionControlMaskNone);
        state.module.AddFunction(spv::OpBranchConditional, failed, failedLabel, next);
        EmitLabel(state, failedLabel);
        RecordBdaFaultReason(state, address, bytes, instruction, reason);
        state.module.AddFunction(spv::OpReturnValue, BdaConstant(state, 0u));
        EmitLabel(state, next);
    };
    const auto length = state.module.AllocateId();
    state.module.AddFunction(spv::OpArrayLength, u32, length, state.bdaPagetableVariable, 0u);
    fail(binary(spv::OpULessThan, boolean, length, constant(4)), BdaAbi::FaultReason::InvalidTable);
    const auto count = BdaLoadWord(state, constant(1));
    const auto available = binary(spv::OpISub, u32, length, constant(4));
    const auto end = binary(spv::OpIAdd, u64, address, Unary(state, spv::OpUConvert, u64, bytes));
    failFirst({
        {binary(spv::OpINotEqual, boolean, BdaLoadWord(state, constant(0)), constant(BdaAbi::Version)), BdaAbi::FaultReason::InvalidTable},
        {binary(spv::OpINotEqual, boolean, BdaLoadWord(state, constant(2)), constant(sizeof(BdaAbi::Range))), BdaAbi::FaultReason::InvalidTable},
        {binary(spv::OpINotEqual, boolean, BdaLoadWord(state, constant(3)), constant(0)), BdaAbi::FaultReason::InvalidTable},
        {binary(spv::OpINotEqual, boolean, binary(spv::OpBitwiseAnd, u32, available, constant(7)), constant(0)), BdaAbi::FaultReason::InvalidTable},
        {binary(spv::OpINotEqual, boolean, count, binary(spv::OpShiftRightLogical, u32, available, constant(3))), BdaAbi::FaultReason::InvalidTable},
        {binary(spv::OpIEqual, boolean, bytes, constant(0)), BdaAbi::FaultReason::Overflow},
        {binary(spv::OpULessThanEqual, boolean, end, address), BdaAbi::FaultReason::Overflow},
    });
    const auto header = state.module.AllocateId();
    const auto body = state.module.AllocateId();
    const auto continuation = state.module.AllocateId();
    const auto merge = state.module.AllocateId();
    const auto entryLabel = state.currentLabel;
    state.module.AddFunction(spv::OpBranch, header);
    EmitLabel(state, header);
    const auto lower = state.module.AllocateId();
    const auto upper = state.module.AllocateId();
    const auto newLow = state.module.AllocateId();
    const auto newHigh = state.module.AllocateId();
    state.module.AddFunction(spv::OpPhi, u32, lower, constant(0), entryLabel, newLow, continuation);
    state.module.AddFunction(spv::OpPhi, u32, upper, count, entryLabel, newHigh, continuation);
    const auto search = binary(spv::OpULessThan, boolean, lower, upper);
    state.module.AddFunction(spv::OpLoopMerge, merge, continuation, spv::LoopControlMaskNone);
    state.module.AddFunction(spv::OpBranchConditional, search, body, merge);
    EmitLabel(state, body);
    const auto midpoint = binary(spv::OpIAdd, u32, lower, binary(spv::OpShiftRightLogical, u32, binary(spv::OpISub, u32, upper, lower), constant(1)));
    const auto entry = binary(spv::OpIAdd, u32, constant(4), binary(spv::OpIMul, u32, midpoint, constant(8)));
    const auto before = binary(spv::OpULessThan, boolean, address, BdaLoadAddress(state, entry));
    state.module.AddFunction(spv::OpSelect, u32, newLow, before, lower, binary(spv::OpIAdd, u32, midpoint, constant(1)));
    state.module.AddFunction(spv::OpSelect, u32, newHigh, before, midpoint, upper);
    state.module.AddFunction(spv::OpBranch, continuation);
    EmitLabel(state, continuation);
    state.module.AddFunction(spv::OpBranch, header);
    EmitLabel(state, merge);
    const auto index = lower;
    fail(binary(spv::OpIEqual, boolean, index, constant(0)), BdaAbi::FaultReason::Unmapped);
    const auto selected = binary(spv::OpIAdd, u32, constant(4), binary(spv::OpIMul, u32, binary(spv::OpISub, u32, index, constant(1)), constant(8)));
    const auto at = [&](std::uint32_t offset) { return binary(spv::OpIAdd, u32, selected, constant(offset)); };
    const auto begin = BdaLoadAddress(state, selected);
    const auto finish = BdaLoadAddress(state, at(2));
    const auto base = BdaLoadAddress(state, at(4));
    const auto permissions = BdaLoadWord(state, at(6));
    const auto offset = binary(spv::OpISub, u64, address, begin);
    const auto result = binary(spv::OpIAdd, u64, base, offset);
    const auto deviceEnd = binary(spv::OpIAdd, u64, result, Unary(state, spv::OpUConvert, u64, bytes));
    failFirst({
        {binary(spv::OpINotEqual, boolean, BdaLoadWord(state, at(7)), constant(0)), BdaAbi::FaultReason::InvalidTable},
        {binary(spv::OpUGreaterThanEqual, boolean, begin, finish), BdaAbi::FaultReason::InvalidTable},
        {binary(spv::OpUGreaterThan, boolean, end, finish), BdaAbi::FaultReason::Unmapped},
        {binary(spv::OpINotEqual, boolean, binary(spv::OpBitwiseAnd, u32, permissions, constant(permission)), constant(permission)), BdaAbi::FaultReason::Permission},
        {binary(spv::OpIEqual, boolean, base, BdaConstant(state, 0)), BdaAbi::FaultReason::InvalidTable},
        {binary(spv::OpULessThan, boolean, result, base), BdaAbi::FaultReason::Overflow},
        {binary(spv::OpULessThanEqual, boolean, deviceEnd, result), BdaAbi::FaultReason::Overflow},
    });
    state.module.AddFunction(spv::OpReturnValue, result);
    state.module.AddFunction(spv::OpFunctionEnd);
    return function;
}

}

static std::uint32_t DefineBdaNoteWrite(SpirvEmitterState& state) {
    const auto u32 = TypeU32(state);
    const auto boolean = TypeBool(state);
    const auto constant = [&](std::uint32_t value) { return ConstantU32(state, value); };
    const auto binary = [&](std::uint32_t op, std::uint32_t type, std::uint32_t left, std::uint32_t right) { return Binary(state, op, type, left, right); };
    const auto voidType = state.module.Type(spv::OpTypeVoid);
    const auto function = state.module.AllocateId();
    state.module.AddName(function, "note_bda_write");
    state.module.AddFunction(spv::OpFunction, voidType, function, spv::FunctionControlMaskNone, state.module.Type(spv::OpTypeFunction, voidType, TypeScalarU64(state)));
    const auto address = state.module.AllocateId();
    state.module.AddFunction(spv::OpFunctionParameter, TypeScalarU64(state), address);
    EmitLabel(state, state.module.AllocateId());
    const auto done = state.module.AllocateId();
    state.module.AddFunction(spv::OpVariable, TypePointer(state, spv::StorageClassFunction, boolean), done, spv::StorageClassFunction);
    const auto counter = state.module.AllocateId();
    state.module.AddFunction(spv::OpVariable, TypePointer(state, spv::StorageClassFunction, u32), counter, spv::StorageClassFunction);
    state.module.AddFunction(spv::OpStore, done, ConstantBool(state, false));
    state.module.AddFunction(spv::OpStore, counter, constant(0u));
    const auto u64 = TypeScalarU64(state);
    const auto key = binary(spv::OpIAdd, u32, Unary(state, spv::OpUConvert, u32, binary(spv::OpShiftRightLogical, u64, address, BdaConstant(state, BdaAbi::WrittenChunkShift))), constant(1));
    const auto page = binary(spv::OpBitwiseAnd, u32, Unary(state, spv::OpUConvert, u32, binary(spv::OpShiftRightLogical, u64, address, BdaConstant(state, BdaAbi::WrittenPageShift))), constant(BdaAbi::WrittenChunkPages - 1u));
    const auto bit = binary(spv::OpShiftLeftLogical, u32, constant(1u), page);
    const auto hash = binary(spv::OpShiftRightLogical, u32, binary(spv::OpIMul, u32, key, constant(0x9e3779b1u)), constant(32u - static_cast<std::uint32_t>(std::countr_zero(BdaAbi::WrittenPageSlots))));
    const auto scope = constant(spv::ScopeDevice);
    const auto relaxed = constant(spv::MemorySemanticsMaskNone);
    const auto header = state.module.AllocateId();
    const auto body = state.module.AllocateId();
    const auto continuation = state.module.AllocateId();
    const auto merge = state.module.AllocateId();
    state.module.AddFunction(spv::OpBranch, header);
    EmitLabel(state, header);
    const auto probe = state.module.AllocateId();
    state.module.AddFunction(spv::OpLoad, u32, probe, counter);
    const auto found = state.module.AllocateId();
    state.module.AddFunction(spv::OpLoad, boolean, found, done);
    const auto searching = binary(spv::OpLogicalAnd, boolean, Unary(state, spv::OpLogicalNot, boolean, found), binary(spv::OpULessThan, boolean, probe, constant(BdaAbi::WrittenPageProbes)));
    state.module.AddFunction(spv::OpLoopMerge, merge, continuation, spv::LoopControlMaskNone);
    state.module.AddFunction(spv::OpBranchConditional, searching, body, merge);
    EmitLabel(state, body);
    const auto slot = binary(spv::OpBitwiseAnd, u32, binary(spv::OpIAdd, u32, hash, probe), constant(BdaAbi::WrittenPageSlots - 1u));
    const auto keyWord = binary(spv::OpIAdd, u32, constant(BdaAbi::WrittenSlotsWord), binary(spv::OpShiftLeftLogical, u32, slot, constant(1u)));
    const auto current = state.module.AllocateId();
    state.module.AddFunction(spv::OpAtomicLoad, u32, current, BdaWord(state, state.faultBufferVariable, keyWord), scope, relaxed);
    const auto same = binary(spv::OpIEqual, boolean, current, key);
    state.module.AddFunction(spv::OpStore, done, same);
    EmitIfCondition(state, Unary(state, spv::OpLogicalNot, boolean, same), [&] {
        const auto previous = state.module.AllocateId();
        state.module.AddFunction(spv::OpAtomicCompareExchange, u32, previous, BdaWord(state, state.faultBufferVariable, keyWord), scope, relaxed, relaxed, key, constant(0u));
        const auto claimed = binary(spv::OpIEqual, boolean, previous, constant(0u));
        state.module.AddFunction(spv::OpStore, done, binary(spv::OpLogicalOr, boolean, claimed, binary(spv::OpIEqual, boolean, previous, key)));
        EmitIfCondition(state, claimed, [&] {
            const auto index = state.module.AllocateId();
            state.module.AddFunction(spv::OpAtomicIAdd, u32, index, BdaWord(state, state.faultBufferVariable, constant(BdaAbi::WrittenCountWord)), scope, relaxed, constant(1u));
            state.module.AddFunction(spv::OpAtomicStore, BdaWord(state, state.faultBufferVariable, binary(spv::OpIAdd, u32, constant(BdaAbi::WrittenListWord), index)), scope, relaxed, slot);
        });
    });
    const auto holds = state.module.AllocateId();
    state.module.AddFunction(spv::OpLoad, boolean, holds, done);
    EmitIfCondition(state, holds, [&] {
        const auto maskWord = binary(spv::OpIAdd, u32, keyWord, constant(1u));
        const auto mask = state.module.AllocateId();
        state.module.AddFunction(spv::OpAtomicLoad, u32, mask, BdaWord(state, state.faultBufferVariable, maskWord), scope, relaxed);
        EmitIfCondition(state, binary(spv::OpIEqual, boolean, binary(spv::OpBitwiseAnd, u32, mask, bit), constant(0u)), [&] {
            state.module.AddFunction(spv::OpAtomicOr, u32, state.module.AllocateId(), BdaWord(state, state.faultBufferVariable, maskWord), scope, relaxed, bit);
        });
    });
    state.module.AddFunction(spv::OpBranch, continuation);
    EmitLabel(state, continuation);
    state.module.AddFunction(spv::OpStore, counter, binary(spv::OpIAdd, u32, probe, constant(1u)));
    state.module.AddFunction(spv::OpBranch, header);
    EmitLabel(state, merge);
    const auto noted = state.module.AllocateId();
    state.module.AddFunction(spv::OpLoad, boolean, noted, done);
    EmitIfCondition(state, Unary(state, spv::OpLogicalNot, boolean, noted), [&] {
        state.module.AddFunction(spv::OpAtomicStore, BdaWord(state, state.faultBufferVariable, constant(BdaAbi::WrittenOverflowWord)), scope, relaxed, constant(1u));
    });
    state.module.AddFunction(spv::OpReturn);
    state.module.AddFunction(spv::OpFunctionEnd);
    return function;
}

void DefineGetBdaPointer(SpirvEmitterState& state) {
    if (state.faultBufferVariable != 0) DefineBdaFaultFunction(state);
    if (!state.program.Info().usesDma) return;
    state.bdaPointerFunction = DefineBdaLookup(state, "get_bda_pointer", true);
    if (!BdaByteReadsForced()) state.bdaProbeFunction = DefineBdaLookup(state, "probe_bda_pointer", false);
    DefineBdaDwordReadFunctions(state);
    DefineBdaSpanReadFunctions(state);
    if (state.program.Info().bdaWrites) {
        state.bdaWritePointerFunction = DefineBdaLookup(state, "get_bda_write_pointer", true, BdaAbi::Write);
        if (!BdaByteReadsForced()) state.bdaWriteProbeFunction = DefineBdaLookup(state, "probe_bda_write_pointer", false, BdaAbi::Write);
        state.bdaAtomicPointerFunction = DefineBdaLookup(state, "get_bda_atomic_pointer", true, BdaAbi::Read | BdaAbi::Write);
        state.bdaNoteWriteFunction = DefineBdaNoteWrite(state);
        DefineBdaByteWriteFunctions(state);
    }
}

}
