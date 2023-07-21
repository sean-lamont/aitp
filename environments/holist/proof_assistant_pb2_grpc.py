# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
"""Client and server classes corresponding to protobuf-defined services."""
import grpc

from environments.holist import proof_assistant_pb2 as deepmath_dot_proof__assistant_dot_proof__assistant__pb2


class ProofAssistantServiceStub(object):
    """Missing associated documentation comment in .proto file."""

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.ApplyTactic = channel.unary_unary(
                '/deepmath.ProofAssistantService/ApplyTactic',
                request_serializer=deepmath_dot_proof__assistant_dot_proof__assistant__pb2.ApplyTacticRequest.SerializeToString,
                response_deserializer=deepmath_dot_proof__assistant_dot_proof__assistant__pb2.ApplyTacticResponse.FromString,
                )
        self.VerifyProof = channel.unary_unary(
                '/deepmath.ProofAssistantService/VerifyProof',
                request_serializer=deepmath_dot_proof__assistant_dot_proof__assistant__pb2.VerifyProofRequest.SerializeToString,
                response_deserializer=deepmath_dot_proof__assistant_dot_proof__assistant__pb2.VerifyProofResponse.FromString,
                )
        self.RegisterTheorem = channel.unary_unary(
                '/deepmath.ProofAssistantService/RegisterTheorem',
                request_serializer=deepmath_dot_proof__assistant_dot_proof__assistant__pb2.RegisterTheoremRequest.SerializeToString,
                response_deserializer=deepmath_dot_proof__assistant_dot_proof__assistant__pb2.RegisterTheoremResponse.FromString,
                )


class ProofAssistantServiceServicer(object):
    """Missing associated documentation comment in .proto file."""

    def ApplyTactic(self, request, context):
        """Apply a tactic to a goal, potentially generating new subgoals.
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def VerifyProof(self, request, context):
        """Verify that a sequence of tactics proves a goal using the proof assistant.
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def RegisterTheorem(self, request, context):
        """Register a new theorem with the proof assistant and verify/generate
        its fingerprint. The theorem will be assumed as one that can be used
        as a valid premise for later proofs.
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_ProofAssistantServiceServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'ApplyTactic': grpc.unary_unary_rpc_method_handler(
                    servicer.ApplyTactic,
                    request_deserializer=deepmath_dot_proof__assistant_dot_proof__assistant__pb2.ApplyTacticRequest.FromString,
                    response_serializer=deepmath_dot_proof__assistant_dot_proof__assistant__pb2.ApplyTacticResponse.SerializeToString,
            ),
            'VerifyProof': grpc.unary_unary_rpc_method_handler(
                    servicer.VerifyProof,
                    request_deserializer=deepmath_dot_proof__assistant_dot_proof__assistant__pb2.VerifyProofRequest.FromString,
                    response_serializer=deepmath_dot_proof__assistant_dot_proof__assistant__pb2.VerifyProofResponse.SerializeToString,
            ),
            'RegisterTheorem': grpc.unary_unary_rpc_method_handler(
                    servicer.RegisterTheorem,
                    request_deserializer=deepmath_dot_proof__assistant_dot_proof__assistant__pb2.RegisterTheoremRequest.FromString,
                    response_serializer=deepmath_dot_proof__assistant_dot_proof__assistant__pb2.RegisterTheoremResponse.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'deepmath.ProofAssistantService', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))


 # This class is part of an EXPERIMENTAL API.
class ProofAssistantService(object):
    """Missing associated documentation comment in .proto file."""

    @staticmethod
    def ApplyTactic(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/deepmath.ProofAssistantService/ApplyTactic',
            deepmath_dot_proof__assistant_dot_proof__assistant__pb2.ApplyTacticRequest.SerializeToString,
            deepmath_dot_proof__assistant_dot_proof__assistant__pb2.ApplyTacticResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def VerifyProof(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/deepmath.ProofAssistantService/VerifyProof',
            deepmath_dot_proof__assistant_dot_proof__assistant__pb2.VerifyProofRequest.SerializeToString,
            deepmath_dot_proof__assistant_dot_proof__assistant__pb2.VerifyProofResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def RegisterTheorem(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/deepmath.ProofAssistantService/RegisterTheorem',
            deepmath_dot_proof__assistant_dot_proof__assistant__pb2.RegisterTheoremRequest.SerializeToString,
            deepmath_dot_proof__assistant_dot_proof__assistant__pb2.RegisterTheoremResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)
