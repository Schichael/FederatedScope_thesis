# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: gRPC_comm_manager.proto
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()

DESCRIPTOR = _descriptor.FileDescriptor(
    name='gRPC_comm_manager.proto',
    package='',
    syntax='proto3',
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
    serialized_pb=
    b'\n\x17gRPC_comm_manager.proto\"\x1d\n\x0eMessageRequest\x12\x0b\n\x03msg\x18\x01 \x01(\t\"\x1e\n\x0fMessageResponse\x12\x0b\n\x03msg\x18\x01 \x01(\t2D\n\x10gRPCComServeFunc\x12\x30\n\x0bsendMessage\x12\x0f.MessageRequest\x1a\x10.MessageResponseb\x06proto3'
)

_MESSAGEREQUEST = _descriptor.Descriptor(
    name='MessageRequest',
    full_name='MessageRequest',
    filename=None,
    file=DESCRIPTOR,
    containing_type=None,
    create_key=_descriptor._internal_create_key,
    fields=[
        _descriptor.FieldDescriptor(
            name='msg',
            full_name='MessageRequest.msg',
            index=0,
            number=1,
            type=9,
            cpp_type=9,
            label=1,
            has_default_value=False,
            default_value=b"".decode('utf-8'),
            message_type=None,
            enum_type=None,
            containing_type=None,
            is_extension=False,
            extension_scope=None,
            serialized_options=None,
            file=DESCRIPTOR,
            create_key=_descriptor._internal_create_key),
    ],
    extensions=[],
    nested_types=[],
    enum_types=[],
    serialized_options=None,
    is_extendable=False,
    syntax='proto3',
    extension_ranges=[],
    oneofs=[],
    serialized_start=27,
    serialized_end=56,
)

_MESSAGERESPONSE = _descriptor.Descriptor(
    name='MessageResponse',
    full_name='MessageResponse',
    filename=None,
    file=DESCRIPTOR,
    containing_type=None,
    create_key=_descriptor._internal_create_key,
    fields=[
        _descriptor.FieldDescriptor(
            name='msg',
            full_name='MessageResponse.msg',
            index=0,
            number=1,
            type=9,
            cpp_type=9,
            label=1,
            has_default_value=False,
            default_value=b"".decode('utf-8'),
            message_type=None,
            enum_type=None,
            containing_type=None,
            is_extension=False,
            extension_scope=None,
            serialized_options=None,
            file=DESCRIPTOR,
            create_key=_descriptor._internal_create_key),
    ],
    extensions=[],
    nested_types=[],
    enum_types=[],
    serialized_options=None,
    is_extendable=False,
    syntax='proto3',
    extension_ranges=[],
    oneofs=[],
    serialized_start=58,
    serialized_end=88,
)

DESCRIPTOR.message_types_by_name['MessageRequest'] = _MESSAGEREQUEST
DESCRIPTOR.message_types_by_name['MessageResponse'] = _MESSAGERESPONSE
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

MessageRequest = _reflection.GeneratedProtocolMessageType(
    'MessageRequest',
    (_message.Message, ),
    {
        'DESCRIPTOR': _MESSAGEREQUEST,
        '__module__': 'gRPC_comm_manager_pb2'
        # @@protoc_insertion_point(class_scope:MessageRequest)
    })
_sym_db.RegisterMessage(MessageRequest)

MessageResponse = _reflection.GeneratedProtocolMessageType(
    'MessageResponse',
    (_message.Message, ),
    {
        'DESCRIPTOR': _MESSAGERESPONSE,
        '__module__': 'gRPC_comm_manager_pb2'
        # @@protoc_insertion_point(class_scope:MessageResponse)
    })
_sym_db.RegisterMessage(MessageResponse)

_GRPCCOMSERVEFUNC = _descriptor.ServiceDescriptor(
    name='gRPCComServeFunc',
    full_name='gRPCComServeFunc',
    file=DESCRIPTOR,
    index=0,
    serialized_options=None,
    create_key=_descriptor._internal_create_key,
    serialized_start=90,
    serialized_end=158,
    methods=[
        _descriptor.MethodDescriptor(
            name='sendMessage',
            full_name='gRPCComServeFunc.sendMessage',
            index=0,
            containing_service=None,
            input_type=_MESSAGEREQUEST,
            output_type=_MESSAGERESPONSE,
            serialized_options=None,
            create_key=_descriptor._internal_create_key,
        ),
    ])
_sym_db.RegisterServiceDescriptor(_GRPCCOMSERVEFUNC)

DESCRIPTOR.services_by_name['gRPCComServeFunc'] = _GRPCCOMSERVEFUNC

# @@protoc_insertion_point(module_scope)
