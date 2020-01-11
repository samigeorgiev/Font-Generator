import React from "react";

import NavigationLink from '../NavigationLink';

const toolbar = props => {
    const links = [
        { to: 'test1', value: 'Test' },
        { to: 'test2', value: 'Test' },
        { to: 'test3', value: 'Test' },
        { to: 'test4', value: 'Test' }
    ];
    return (
        <nav className="Toolbar">
            <ul>
                {links.map(link => <NavigationLink key={link.to} value={link.value} />)}
            </ul>
        </nav>
    );
};

export default toolbar;